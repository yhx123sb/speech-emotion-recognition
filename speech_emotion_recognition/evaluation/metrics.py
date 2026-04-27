"""
评估指标模块
提供：准确率、精确率、召回率、F1分数、混淆矩阵、分类报告、可视化
支持：传统 ML 和深度学习模型的统一评估接口
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)

from ..utils.config import EMOTION_LABELS, NUM_EMOTIONS


class MetricsEvaluator:
    """
    模型评估器
    提供统一的评估接口，支持多种指标计算和可视化
    """

    def __init__(self, emotion_labels: Optional[List[str]] = None):
        """
        初始化评估器

        Args:
            emotion_labels: 情绪标签列表
        """
        self.emotion_labels = emotion_labels or EMOTION_LABELS
        self.num_classes = len(self.emotion_labels)

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算准确率"""
        return accuracy_score(y_true, y_pred)

    def precision(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted",
    ) -> float:
        """计算精确率"""
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted",
    ) -> float:
        """计算召回率"""
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted",
    ) -> float:
        """计算 F1 分数"""
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        labels = np.unique(np.concatenate([y_true, y_pred]))
        return confusion_matrix(y_true, y_pred, labels=labels)

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False,
    ) -> Union[str, Dict]:
        """
        生成分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            output_dict: 是否返回字典格式

        Returns:
            分类报告字符串或字典
        """
        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        present_names = [self.emotion_labels[i] for i in present_labels if i < len(self.emotion_labels)]
        return classification_report(
            y_true,
            y_pred,
            target_names=present_names,
            labels=present_labels,
            zero_division=0,
            output_dict=output_dict,
        )

    def evaluate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算所有主要指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            指标字典
        """
        return {
            "accuracy": self.accuracy(y_true, y_pred),
            "precision_weighted": self.precision(y_true, y_pred),
            "recall_weighted": self.recall(y_true, y_pred),
            "f1_weighted": self.f1(y_true, y_pred),
            "precision_macro": self.precision(y_true, y_pred, average="macro"),
            "recall_macro": self.recall(y_true, y_pred, average="macro"),
            "f1_macro": self.f1(y_true, y_pred, average="macro"),
        }

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (10, 8),
        normalize: bool = True,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题
            figsize: 图表尺寸
            normalize: 是否归一化
            save_path: 保存路径
            show: 是否显示图表

        Returns:
            matplotlib Figure 对象
        """
        cm = self.confusion_matrix(y_true, y_pred)
        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        present_names = [self.emotion_labels[i] for i in present_labels if i < len(self.emotion_labels)]

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            vmin, vmax = 0, 1
        else:
            fmt = "d"
            vmin, vmax = 0, cm.max()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=present_names,
            yticklabels=present_names,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"[保存] 混淆矩阵已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Classification Report",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制分类报告热力图

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib Figure 对象
        """
        report = self.classification_report(y_true, y_pred, output_dict=True)

        # 提取每个类别的指标
        metrics = ["precision", "recall", "f1-score"]
        data = {}
        for emotion in self.emotion_labels:
            if emotion in report:
                data[emotion] = [report[emotion][m] for m in metrics]

        # 添加平均值
        data["macro avg"] = [report["macro avg"][m] for m in metrics]
        data["weighted avg"] = [report["weighted avg"][m] for m in metrics]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            np.array(list(data.values())),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=metrics,
            yticklabels=list(data.keys()),
            vmin=0,
            vmax=1,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        绘制训练历史曲线

        Args:
            history: 训练历史字典（含 train_loss, val_loss, train_acc, val_acc）
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 损失曲线
        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train Loss", alpha=0.8)
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Val Loss", alpha=0.8)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        if "train_acc" in history:
            axes[1].plot(history["train_acc"], label="Train Acc", alpha=0.8)
        if "val_acc" in history:
            axes[1].plot(history["val_acc"], label="Val Acc", alpha=0.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, np.ndarray]],
        title: str = "Model Comparison",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        比较多个模型的性能

        Args:
            model_results: 模型结果字典
                {model_name: {"y_true": ..., "y_pred": ...}}
            title: 图表标题
            figsize: 图表尺寸
            save_path: 保存路径
            show: 是否显示

        Returns:
            matplotlib Figure 对象
        """
        model_names = list(model_results.keys())
        metrics_data = {
            "Accuracy": [],
            "Precision (weighted)": [],
            "Recall (weighted)": [],
            "F1-Score (weighted)": [],
        }

        for name in model_names:
            y_true = model_results[name]["y_true"]
            y_pred = model_results[name]["y_pred"]
            metrics_data["Accuracy"].append(self.accuracy(y_true, y_pred))
            metrics_data["Precision (weighted)"].append(
                self.precision(y_true, y_pred)
            )
            metrics_data["Recall (weighted)"].append(self.recall(y_true, y_pred))
            metrics_data["F1-Score (weighted)"].append(self.f1(y_true, y_pred))

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(model_names))
        bar_width = 0.2
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            bars = ax.bar(
                x + i * bar_width,
                values,
                bar_width,
                label=metric_name,
                color=colors[i],
                alpha=0.85,
            )
            # 在柱状图上标注数值
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def print_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ):
        """
        打印评估结果摘要

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
        """
        metrics = self.evaluate_all(y_true, y_pred)

        print(f"\n{'='*60}")
        print(f"  {model_name} - 评估结果")
        print(f"{'='*60}")
        print(f"  准确率 (Accuracy):        {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision, w):    {metrics['precision_weighted']:.4f}")
        print(f"  召回率 (Recall, w):       {metrics['recall_weighted']:.4f}")
        print(f"  F1 分数 (F1, w):          {metrics['f1_weighted']:.4f}")
        print(f"  ─────────────────────────────────────────")
        print(f"  精确率 (Precision, macro): {metrics['precision_macro']:.4f}")
        print(f"  召回率 (Recall, macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 分数 (F1, macro):       {metrics['f1_macro']:.4f}")
        print(f"{'='*60}")

    def print_confusion_matrix_text(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        以文本形式打印混淆矩阵，只显示实际出现的类别
        """
        cm = self.confusion_matrix(y_true, y_pred)
        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        present_names = [self.emotion_labels[i] for i in present_labels if i < len(self.emotion_labels)]

        print(f"\n  混淆矩阵:")
        print(f"  {chr(39)}{chr(39)}:>12", end="")
        for label in present_names:
            print(f"{label:>10}", end="")
        print()

        for i, label in enumerate(present_names):
            print(f"  {label:>10}:", end="")
            for j in range(len(present_names)):
                print(f"{cm[i, j]:>10}", end="")
            print()

def evaluate_model_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    history: Optional[Dict] = None,
    save_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> Dict:
    """
    综合评估模型（一键评估 + 可视化）

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        history: 训练历史（可选，用于绘制学习曲线）
        save_dir: 保存目录
        show_plots: 是否显示图表

    Returns:
        评估指标字典
    """
    evaluator = MetricsEvaluator()

    # 打印评估摘要
    evaluator.print_summary(y_true, y_pred, model_name)

    # 打印分类报告
    report = evaluator.classification_report(y_true, y_pred)
    print(f"\n  分类报告:")
    print(report)

    # 打印混淆矩阵
    evaluator.print_confusion_matrix_text(y_true, y_pred)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存混淆矩阵图
        evaluator.plot_confusion_matrix(
            y_true, y_pred,
            title=f"{model_name} - Confusion Matrix",
            save_path=save_dir / f"{model_name.lower()}_confusion_matrix.png",
            show=show_plots,
        )

        # 保存分类报告图
        evaluator.plot_classification_report(
            y_true, y_pred,
            title=f"{model_name} - Classification Report",
            save_path=save_dir / f"{model_name.lower()}_classification_report.png",
            show=show_plots,
        )

        # 如果有训练历史，保存学习曲线
        if history:
            evaluator.plot_training_history(
                history,
                title=f"{model_name} - Training History",
                save_path=save_dir / f"{model_name.lower()}_training_history.png",
                show=show_plots,
            )

    # 返回详细指标
    metrics = evaluator.evaluate_all(y_true, y_pred)
    metrics["classification_report"] = evaluator.classification_report(
        y_true, y_pred, output_dict=True
    )
    metrics["confusion_matrix"] = evaluator.confusion_matrix(y_true, y_pred)

    return metrics


def compare_and_report(
    model_results: Dict[str, Dict[str, np.ndarray]],
    save_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    """
    比较多个模型并生成报告表格

    Args:
        model_results: {model_name: {"y_true": ..., "y_pred": ...}}
        save_dir: 保存目录
        show_plots: 是否显示图表

    Returns:
        包含各模型指标的 DataFrame
    """
    import pandas as pd

    evaluator = MetricsEvaluator()
    comparison_data = []

    for model_name, results in model_results.items():
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        metrics = evaluator.evaluate_all(y_true, y_pred)
        metrics["Model"] = model_name
        comparison_data.append(metrics)

    df = pd.DataFrame(comparison_data)
    df = df.set_index("Model")

    print(f"\n{'='*70}")
    print(f"  模型对比结果")
    print(f"{'='*70}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*70}")

    # 绘制对比图
    evaluator.compare_models(
        model_results,
        title="Model Performance Comparison",
        save_path=save_dir / "model_comparison.png" if save_dir else None,
        show=show_plots,
    )

    return df


# 尝试导入 pandas（可选依赖）
try:
    import pandas as pd
except ImportError:
    pd = None
    def compare_and_report(model_results, save_dir=None, show_plots=True):
        """如果 pandas 不可用，使用简单比较"""
        print("[警告] pandas 未安装，使用简化比较模式")
        evaluator = MetricsEvaluator()
        for model_name, results in model_results.items():
            y_true = results["y_true"]
            y_pred = results["y_pred"]
            evaluator.print_summary(y_true, y_pred, model_name)
        return None
