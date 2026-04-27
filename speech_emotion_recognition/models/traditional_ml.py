"""
传统机器学习模型
支持：SVM (Support Vector Machine) 和 Random Forest
基于122维聚合特征向量进行情感分类
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.config import (
    SAVED_MODELS_DIR,
    EMOTION_LABELS,
    NUM_EMOTIONS,
    training_cfg,
)


class SVMClassifier:
    """
    SVM 情感分类器
    使用 RBF 核的 SVM，适合高维特征的小样本分类
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        """
        初始化 SVM 分类器

        Args:
            kernel: 核函数 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数（越大越严格拟合）
            gamma: RBF 核的系数
            class_weight: 类别权重策略
            random_state: 随机种子
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

    def _build_model(self) -> SVC:
        """构建 SVM 模型"""
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            random_state=self.random_state,
            probability=True,  # 启用概率输出
            verbose=False,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        训练 SVM 模型

        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)

        Returns:
            训练历史（准确率等）
        """
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 构建并训练模型
        self.model = self._build_model()
        print(f"[SVM] 开始训练 (kernel={self.kernel}, C={self.C})...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # 评估训练集
        train_acc = self.model.score(X_train_scaled, y_train)
        result = {"train_accuracy": train_acc}

        # 评估验证集
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_acc = self.model.score(X_val_scaled, y_val)
            result["val_accuracy"] = val_acc
            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")
        else:
            print(f"  训练准确率: {train_acc:.4f}")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征数组

        Returns:
            预测标签
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数组

        Returns:
            概率矩阵 (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
    ) -> Dict:
        """
        网格搜索超参数

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格，默认搜索常用参数
            cv: 交叉验证折数

        Returns:
            最佳参数和得分
        """
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "kernel": ["rbf", "linear"],
            }

        X_scaled = self.scaler.fit_transform(X_train)
        base_model = SVC(class_weight="balanced", random_state=self.random_state, probability=True)

        print(f"[SVM GridSearch] 搜索参数空间...")
        print(f"  参数网格: {param_grid}")
        print(f"  交叉验证: {cv}折")

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_scaled, y_train)

        # 更新最佳参数
        self.C = grid.best_params_.get("C", self.C)
        self.gamma = grid.best_params_.get("gamma", self.gamma)
        self.kernel = grid.best_params_.get("kernel", self.kernel)

        print(f"\n[最佳参数] {grid.best_params_}")
        print(f"[最佳得分] {grid.best_score_:.4f}")

        return {"best_params": grid.best_params_, "best_score": grid.best_score_}

    def save(self, filepath: Optional[Path] = None) -> str:
        """
        保存模型

        Args:
            filepath: 保存路径，默认为 models/saved/svm_model.pkl

        Returns:
            保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")

        if filepath is None:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = SAVED_MODELS_DIR / "svm_model.pkl"

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "kernel": self.kernel,
                "C": self.C,
                "gamma": self.gamma,
                "emotion_labels": EMOTION_LABELS,
                "num_classes": NUM_EMOTIONS,
            },
            str(filepath),
        )
        print(f"[保存] SVM 模型已保存到: {filepath}")
        return str(filepath)

    def load(self, filepath: Path) -> bool:
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            是否成功
        """
        if not filepath.exists():
            print(f"[错误] 模型文件不存在: {filepath}")
            return False

        data = joblib.load(str(filepath))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.kernel = data["kernel"]
        self.C = data["C"]
        self.gamma = data["gamma"]
        self.is_trained = True
        print(f"[加载] SVM 模型已加载: {filepath}")
        return True


class RandomForestClassifier:
    """
    Random Forest 情感分类器
    集成学习方法，对噪声鲁棒性强，适合高维特征
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        """
        初始化 Random Forest 分类器

        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度（None=不限制）
            min_samples_split: 内部节点再划分所需最小样本数
            min_samples_leaf: 叶子节点最少样本数
            class_weight: 类别权重
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_importances_ = None

    def _build_model(self):
        """构建 Random Forest 模型"""
        return SklearnRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=False,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        训练 Random Forest 模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练历史
        """
        # Random Forest 不需要标准化，但保持一致性
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = self._build_model()
        print(f"[RandomForest] 开始训练 (n_estimators={self.n_estimators})...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_

        # 评估训练集
        train_acc = self.model.score(X_train_scaled, y_train)
        result = {"train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_acc = self.model.score(X_val_scaled, y_val)
            result["val_accuracy"] = val_acc
            print(f"  训练准确率: {train_acc:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")
        else:
            print(f"  训练准确率: {train_acc:.4f}")

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train()")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
    ) -> Dict:
        """
        网格搜索超参数

        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数

        Returns:
            最佳参数和得分
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }

        X_scaled = self.scaler.fit_transform(X_train)
        base_model = SklearnRandomForest(
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )

        print(f"[RF GridSearch] 搜索参数空间...")
        print(f"  参数网格: {param_grid}")
        print(f"  交叉验证: {cv}折")

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_scaled, y_train)

        # 更新最佳参数
        self.n_estimators = grid.best_params_.get("n_estimators", self.n_estimators)
        self.max_depth = grid.best_params_.get("max_depth", self.max_depth)
        self.min_samples_split = grid.best_params_.get("min_samples_split", self.min_samples_split)
        self.min_samples_leaf = grid.best_params_.get("min_samples_leaf", self.min_samples_leaf)

        print(f"\n[最佳参数] {grid.best_params_}")
        print(f"[最佳得分] {grid.best_score_:.4f}")

        return {"best_params": grid.best_params_, "best_score": grid.best_score_}

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict:
        """
        获取特征重要性

        Args:
            feature_names: 特征名称列表

        Returns:
            特征重要性字典
        """
        if self.feature_importances_ is None:
            raise ValueError("模型尚未训练，无法获取特征重要性")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        importance_dict = {
            name: imp for name, imp in zip(feature_names, self.feature_importances_)
        }
        # 按重要性排序
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        return importance_dict

    def save(self, filepath: Optional[Path] = None) -> str:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")

        if filepath is None:
            SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = SAVED_MODELS_DIR / "random_forest_model.pkl"

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "feature_importances": self.feature_importances_,
                "emotion_labels": EMOTION_LABELS,
                "num_classes": NUM_EMOTIONS,
            },
            str(filepath),
        )
        print(f"[保存] RandomForest 模型已保存到: {filepath}")
        return str(filepath)

    def load(self, filepath: Path) -> bool:
        """加载模型"""
        if not filepath.exists():
            print(f"[错误] 模型文件不存在: {filepath}")
            return False

        data = joblib.load(str(filepath))
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.n_estimators = data["n_estimators"]
        self.max_depth = data["max_depth"]
        self.feature_importances_ = data["feature_importances"]
        self.is_trained = True
        print(f"[加载] RandomForest 模型已加载: {filepath}")
        return True


def train_and_evaluate_traditional_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_grid_search: bool = False,
) -> Dict[str, object]:
    """
    一站式训练和评估所有传统 ML 模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        use_grid_search: 是否进行网格搜索

    Returns:
        包含所有模型和结果的字典
    """
    results = {}

    # 1. SVM
    print("\n" + "=" * 60)
    print("  训练 SVM 模型")
    print("=" * 60)
    svm = SVMClassifier()

    if use_grid_search:
        svm.grid_search(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    svm.train(X_train, y_train, X_val, y_val)
    svm_pred = svm.predict(X_test)
    svm.save()
    results["svm"] = {
        "model": svm,
        "predictions": svm_pred,
    }

    # 2. Random Forest
    print("\n" + "=" * 60)
    print("  训练 Random Forest 模型")
    print("=" * 60)
    rf = RandomForestClassifier()

    if use_grid_search:
        rf.grid_search(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    rf.train(X_train, y_train, X_val, y_val)
    rf_pred = rf.predict(X_test)
    rf.save()
    results["random_forest"] = {
        "model": rf,
        "predictions": rf_pred,
    }

    return results
