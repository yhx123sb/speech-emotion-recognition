"""
使用 GPU 加速训练所有模型
- SVM, Random Forest (sklearn, CPU)
- MLP (PyTorch, GPU加速)
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_emotion_recognition.utils.config import (
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    EMOTION_LABELS,
    training_cfg,
)
from speech_emotion_recognition.models.traditional_ml import (
    train_and_evaluate_traditional_models,
)
from speech_emotion_recognition.models.cnn_model import EmotionMLP
from speech_emotion_recognition.models.trainer import Trainer
from speech_emotion_recognition.evaluation.metrics import (
    MetricsEvaluator,
    evaluate_model_comprehensive,
)


def load_casia_data():
    """加载 CASIA 数据并连续化标签"""
    data_dir = PROCESSED_DATA_DIR
    prefix = "casia"

    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    data_dict = {}

    print(f"[加载] 从 {data_dir} 加载 CASIA 数据...")
    for name in names:
        filepath = data_dir / f"{prefix}_{name}.npy"
        if filepath.exists():
            data_dict[name] = np.load(str(filepath))
            print(f"  [OK] {name}: {data_dict[name].shape}")

    # 连续化标签
    all_y = np.concatenate([data_dict["y_train"], data_dict["y_val"], data_dict["y_test"]])
    unique_labels = np.sort(np.unique(all_y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    print(f"[标签映射] {label_map}")

    for key in ["y_train", "y_val", "y_test"]:
        data_dict[key] = np.array([label_map[y] for y in data_dict[key]])

    return data_dict, len(unique_labels)


def train_svm(data_dict):
    """训练 SVM (CPU)"""
    print("\n" + "=" * 60)
    print("  训练 SVM 模型...")
    print("=" * 60)

    from speech_emotion_recognition.models.traditional_ml import SVMClassifier
    from sklearn.metrics import accuracy_score, f1_score

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    start = time.time()

    svm = SVMClassifier(kernel="rbf", C=1.0)
    svm.train(X_train, y_train, data_dict["X_val"], data_dict["y_val"])
    y_pred = svm.predict(X_test)

    elapsed = time.time() - start
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"  SVM 测试准确率: {acc:.4f}")
    print(f"  SVM F1分数(weighted): {f1:.4f}")
    print(f"  训练+评估耗时: {elapsed:.2f}s")

    svm.save(SAVED_MODELS_DIR / "svm_model.pkl")

    return {"y_true": y_test, "y_pred": y_pred, "accuracy": acc, "f1": f1}


def train_rf(data_dict):
    """训练 Random Forest (CPU)"""
    print("\n" + "=" * 60)
    print("  训练 Random Forest 模型...")
    print("=" * 60)

    from speech_emotion_recognition.models.traditional_ml import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    start = time.time()

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.train(X_train, y_train, data_dict["X_val"], data_dict["y_val"])
    y_pred = rf.predict(X_test)

    elapsed = time.time() - start
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"  RF 测试准确率: {acc:.4f}")
    print(f"  RF F1分数(weighted): {f1:.4f}")
    print(f"  训练+评估耗时: {elapsed:.2f}s")

    rf.save(SAVED_MODELS_DIR / "random_forest_model.pkl")

    return {"y_true": y_test, "y_pred": y_pred, "accuracy": acc, "f1": f1}


def train_mlp_gpu(data_dict, n_classes):
    """训练 MLP 模型 (GPU 加速)"""
    print("\n" + "=" * 60)
    print("  训练 MLP 模型 (GPU 加速)...")
    print("=" * 60)

    import torch

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    input_dim = X_train.shape[1]
    print(f"  输入维度: {input_dim}")
    print(f"  类别数: {n_classes}")
    print(f"  训练样本: {len(X_train)}, 验证样本: {len(X_val)}, 测试样本: {len(X_test)}")

    # 创建模型
    model = EmotionMLP(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[512, 256, 128],  # 更大的网络，利用 GPU
        dropout_rate=0.3,
    )

    # 创建训练器 (自动使用 GPU)
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        weight_decay=1e-4,
    )

    print(f"  训练设备: {trainer.device}")

    # 训练
    start = time.time()
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=training_cfg.batch_size,
        num_epochs=training_cfg.num_epochs,
        verbose=True,
    )
    elapsed = time.time() - start

    # 测试
    y_pred = trainer.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"  MLP 训练结果")
    print(f"{'='*60}")
    print(f"  测试准确率: {acc:.4f}")
    print(f"  F1分数(weighted): {f1:.4f}")
    print(f"  训练耗时: {elapsed:.2f}s")
    print(f"  最佳验证准确率: {max(history['val_acc']):.4f}")
    print(f"  训练 epoch 数: {len(history['train_loss'])}")

    # 保存模型
    trainer.save_model(SAVED_MODELS_DIR / "mlp_model.pt")

    return {
        "trainer": trainer,
        "y_true": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "f1": f1,
        "history": history,
    }


def main():
    """使用 GPU 加速训练"""
    print("=" * 60)
    print("  GPU 加速训练 - 语音情绪识别模型")
    print("=" * 60)

    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    else:
        print("  [注意] 未检测到 GPU，将使用 CPU 训练")

    # 创建保存目录
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    data_dict, n_classes = load_casia_data()

    # 保存结果
    all_results = {}

    # 1. SVM
    all_results["svm"] = train_svm(data_dict)

    # 2. Random Forest
    all_results["rf"] = train_rf(data_dict)

    # 3. MLP (GPU 加速)
    all_results["mlp"] = train_mlp_gpu(data_dict, n_classes)

    # 汇总对比
    print("\n" + "=" * 70)
    print("  模型对比汇总")
    print("=" * 70)
    print(f"  {'模型':<15} {'准确率':<12} {'F1分数':<15} {'平台':<12}")
    print(f"  {'─'*54}")
    for name, res in all_results.items():
        device = "GPU" if name == "mlp" and torch.cuda.is_available() else "CPU"
        print(f"  {name:<15} {res['accuracy']:.4f}      {res['f1']:.4f}         {device:<12}")

    print("=" * 70)
    print("  训练完成！")
    print(f"  模型已保存至: {SAVED_MODELS_DIR}")


if __name__ == "__main__":
    main()
