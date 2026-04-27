"""
深度学习训练器
负责：模型训练循环、验证、早停、学习率调度、模型保存/加载
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime

from ..utils.config import SAVED_MODELS_DIR, training_cfg, EMOTION_LABELS, NUM_EMOTIONS


class MetricTracker:
    """训练指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.best_loss = float("inf")
        self.best_accuracy = 0.0

    def update(self, loss: float, accuracy: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        if loss < self.best_loss:
            self.best_loss = loss
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

    def get_average(self) -> Tuple[float, float]:
        if not self.losses:
            return 0.0, 0.0
        return np.mean(self.losses[-100:]), np.mean(self.accuracies[-100:])

    def get_latest(self) -> Tuple[float, float]:
        if not self.losses:
            return 0.0, 0.0
        return self.losses[-1], self.accuracies[-1]


class EarlyStopping:
    """
    早停机制
    当验证集损失不再下降时停止训练，防止过拟合
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best: bool = True):
        """
        Args:
            patience: 容忍的 epochs 数
            min_delta: 最小改善阈值
            restore_best: 是否恢复到最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        检查是否需要早停

        Args:
            val_loss: 验证损失
            model: 当前模型
            epoch: 当前 epoch

        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: nn.Module):
        """恢复到最佳模型状态"""
        if self.restore_best and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"[早停] 恢复到第 {self.best_epoch} 轮的最佳模型 (loss={self.best_loss:.4f})")


class Trainer:
    """
    深度学习训练器
    支持：训练循环、验证、早停、学习率调度、梯度裁剪、模型 checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        optimizer_type: str = "adam",
        scheduler_type: str = "plateau",
    ):
        """
        初始化训练器

        Args:
            model: PyTorch 模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
            optimizer_type: 优化器类型 ('adam', 'sgd', 'adamw')
            scheduler_type: 学习率调度器 ('plateau', 'cosine', 'step', 'none')
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        # 学习率调度器
        if scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=50, eta_min=1e-6
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        else:
            self.scheduler = None

        # 早停
        self.early_stopping = EarlyStopping(
            patience=training_cfg.patience,
            restore_best=True,
        )

        # 指标追踪
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # 梯度裁剪
        self.grad_clip_value = 1.0

        print(f"[训练器] 设备: {self.device}")
        print(f"[训练器] 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[训练器] 优化器: {optimizer_type}, 学习率: {learning_rate}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """
        训练一个 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch 数
            verbose: 是否打印进度

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            if self.grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_value
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            batch_count += 1

            # 打印进度
            if verbose and (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                print(
                    f"  Epoch {epoch:3d} | Batch {batch_idx+1:3d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%"
                )

        avg_loss = total_loss / batch_count
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 100,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> Dict:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批次大小
            num_epochs: 最大训练轮数
            verbose: 是否打印详细信息
            shuffle: 是否打乱训练数据

        Returns:
            训练历史字典
        """
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        print(f"\n{'='*60}")
        print(f"  开始训练 (epochs={num_epochs}, batch_size={batch_size})")
        print(f"{'='*60}")
        print(f"  训练样本: {len(X_train)} | 验证样本: {len(X_val)}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # 训练一个 epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, epoch, verbose=verbose and epoch % 10 == 0
            )

            # 验证
            val_loss, val_acc = self._validate(val_loader)

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rates"].append(current_lr)

            # 更新指标追踪
            self.train_metrics.update(train_loss, train_acc)
            self.val_metrics.update(val_loss, val_acc)

            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 早停检查
            if self.early_stopping(val_loss, self.model, epoch):
                print(f"\n[早停] 验证损失 {self.early_stopping.patience} 轮未改善，停止训练")
                self.early_stopping.restore_best_model(self.model)
                break

        # 恢复最佳模型
        if self.early_stopping.restore_best and self.early_stopping.best_model_state is not None:
            self.early_stopping.restore_best_model(self.model)

        total_time = time.time() - start_time
        best_val_acc = max(self.history["val_acc"])
        best_val_loss = min(self.history["val_loss"])

        print(f"\n{'='*60}")
        print(f"  训练完成！")
        print(f"  总耗时: {total_time:.1f}s")
        print(f"  最佳验证准确率: {best_val_acc:.4f}")
        print(f"  最佳验证损失: {best_val_loss:.4f}")
        print(f"{'='*60}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征数组

        Returns:
            预测标签
        """
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数组

        Returns:
            概率矩阵
        """
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        probabilities = []
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)

    def save_checkpoint(
        self,
        filepath: Optional[Path] = None,
        extra_info: Optional[Dict] = None,
    ) -> str:
        """
        保存模型 checkpoint

        Args:
            filepath: 保存路径
            extra_info: 额外信息

        Returns:
            保存路径
        """
        if filepath is None:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = SAVED_MODELS_DIR / f"cnn_checkpoint_{timestamp}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "emotion_labels": EMOTION_LABELS,
                "num_classes": NUM_EMOTIONS,
            },
        }

        if extra_info:
            checkpoint["extra_info"] = extra_info

        torch.save(checkpoint, str(filepath))
        print(f"[保存] Checkpoint 已保存到: {filepath}")
        return str(filepath)

    def load_checkpoint(self, filepath: Path, load_optimizer: bool = False) -> bool:
        """
        加载 checkpoint

        Args:
            filepath: checkpoint 路径
            load_optimizer: 是否加载优化器状态

        Returns:
            是否成功
        """
        if not filepath.exists():
            print(f"[错误] Checkpoint 不存在: {filepath}")
            return False

        checkpoint = torch.load(str(filepath), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        print(f"[加载] Checkpoint 已加载: {filepath}")
        return True

    def save_model(self, filepath: Optional[Path] = None) -> str:
        """
        保存模型权重（仅模型，不含优化器）

        Args:
            filepath: 保存路径

        Returns:
            保存路径
        """
        if filepath is None:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = SAVED_MODELS_DIR / "cnn_model.pt"

        # 获取模型实际的输出类别数
        try:
            actual_num_classes = self.model.n_classes
        except AttributeError:
            actual_num_classes = NUM_EMOTIONS

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": actual_num_classes,
                "emotion_labels": EMOTION_LABELS,
            },
            str(filepath),
        )
        print(f"[保存] 模型已保存到: {filepath}")
        return str(filepath)


def train_and_evaluate_deep_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    model_name: str = "deep_model",
) -> Tuple[Trainer, Dict]:
    """
    一键训练和评估深度学习模型

    Args:
        model: PyTorch 模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        model_name: 模型名称

    Returns:
        (Trainer 实例, 测试结果)
    """
    print(f"\n{'='*60}")
    print(f"  深度学习模型训练: {model_name}")
    print(f"{'='*60}")

    # 创建训练器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    # 训练
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # 测试评估
    test_pred = trainer.predict(X_test)
    test_acc = np.mean(test_pred == y_test)

    print(f"\n  测试集准确率: {test_acc:.4f}")

    # 保存模型
    trainer.save_model(SAVED_MODELS_DIR / f"{model_name}.pt")

    result = {
        "trainer": trainer,
        "predictions": test_pred,
        "test_accuracy": test_acc,
        "history": history,
    }

    return trainer, result
