"""
数据预处理 Pipeline
负责：批量处理原始音频、提取特征、保存为 numpy 格式、数据集划分
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import pickle

from .audio_loader import AudioLoader
from .dataset_manager import DatasetManager
from ..features.extractor import FeatureExtractor
from ..utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    TARGET_SR,
    TARGET_DURATION,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
    NUM_EMOTIONS,
)


class DataPreprocessor:
    """
    数据预处理器
    处理原始音频 -> 提取特征 -> 保存为 NumPy 格式
    """

    def __init__(
        self,
        audio_loader: Optional[AudioLoader] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        dataset_manager: Optional[DatasetManager] = None,
    ):
        """
        初始化预处理器

        Args:
            audio_loader: 音频加载器
            feature_extractor: 特征提取器
            dataset_manager: 数据集管理器
        """
        self.audio_loader = audio_loader or AudioLoader(
            target_sr=TARGET_SR,
            target_duration=TARGET_DURATION
        )
        self.feature_extractor = feature_extractor or FeatureExtractor(sr=TARGET_SR)
        self.dataset_manager = dataset_manager or DatasetManager()

    def process_ravdess(
        self,
        data_dir: Optional[Path] = None,
        save: bool = True,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        处理 RAVDESS 数据集

        Args:
            data_dir: RAVDESS 数据目录
            save: 是否保存处理后的数据
            augment: 是否进行数据增强

        Returns:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            metadata: 元数据列表
        """
        data_dir = data_dir or RAW_DATA_DIR / "RAVDESS"
        if not data_dir.exists():
            print(f"[错误] RAVDESS 数据目录不存在: {data_dir}")
            print(f"请先运行: python scripts/download_ravdess.py")
            return np.array([]), np.array([]), []

        # 扫描所有音频文件
        samples = self.dataset_manager.organize_ravdess(data_dir)
        if not samples:
            print("[错误] 没有找到音频文件")
            return np.array([]), np.array([]), []

        print(f"\n[信息] 开始处理 {len(samples)} 个音频样本...")

        all_features = []
        all_labels = []
        all_metadata = []
        error_count = 0

        for file_path, emotion, meta in tqdm(samples, desc="处理音频"):
            try:
                # 情绪标签过滤：只保留标准情绪
                if emotion not in EMOTION_TO_IDX:
                    continue

                # 音频预处理
                audio = self.audio_loader.preprocess(file_path)

                # 提取特征向量
                feature_vector = self.feature_extractor.extract_feature_vector(audio)

                all_features.append(feature_vector)
                all_labels.append(EMOTION_TO_IDX[emotion])
                all_metadata.append({
                    "file": file_path,
                    "emotion": emotion,
                    "actor": meta.get("actor", ""),
                    "gender": meta.get("gender", ""),
                    "intensity": meta.get("intensity", ""),
                })

                # 数据增强
                if augment:
                    # 加噪声
                    audio_noisy = self.audio_loader.augment_add_noise(audio)
                    feat_noisy = self.feature_extractor.extract_feature_vector(audio_noisy)
                    all_features.append(feat_noisy)
                    all_labels.append(EMOTION_TO_IDX[emotion])

                    # 音调变化
                    for n_steps in [-2, 2]:
                        audio_pitch = self.audio_loader.augment_pitch_shift(audio, n_steps)
                        feat_pitch = self.feature_extractor.extract_feature_vector(audio_pitch)
                        all_features.append(feat_pitch)
                        all_labels.append(EMOTION_TO_IDX[emotion])

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"\n  [警告] 处理失败: {file_path} -> {e}")

        features = np.array(all_features)
        labels = np.array(all_labels)

        print(f"\n[完成] 处理结果:")
        print(f"  总样本数: {len(features)}")
        print(f"  特征维度: {features.shape[1] if len(features) > 0 else 0}")
        print(f"  失败样本: {error_count}")

        # 打印各类别样本分布
        self._print_class_distribution(labels)

        # 保存数据
        if save and len(features) > 0:
            self._save_processed_data(features, labels, all_metadata, prefix="ravdess")

        return features, labels, all_metadata

    def process_casia(
        self,
        data_dir: Optional[Path] = None,
        save: bool = True,
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        处理 CASIA 中文情感语料库

        Args:
            data_dir: CASIA 数据目录
            save: 是否保存处理后的数据
            augment: 是否进行数据增强

        Returns:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            metadata: 元数据列表
        """
        # 扫描 CASIA 数据
        samples = self.dataset_manager.organize_casia(data_dir)
        if not samples:
            print("[错误] 没有找到 CASIA 音频文件")
            return np.array([]), np.array([]), []

        print(f"\n[信息] 开始处理 {len(samples)} 个 CASIA 音频样本...")

        all_features = []
        all_labels = []
        all_metadata = []
        error_count = 0

        for file_path, emotion, meta in tqdm(samples, desc="处理CASIA音频"):
            try:
                if emotion not in EMOTION_TO_IDX:
                    continue

                # 音频预处理
                audio = self.audio_loader.preprocess(file_path)

                # 提取特征向量
                feature_vector = self.feature_extractor.extract_feature_vector(audio)

                all_features.append(feature_vector)
                all_labels.append(EMOTION_TO_IDX[emotion])
                all_metadata.append({
                    "file": file_path,
                    "emotion": emotion,
                    "actor": meta.get("actor", ""),
                })

                # 数据增强
                if augment:
                    audio_noisy = self.audio_loader.augment_add_noise(audio)
                    feat_noisy = self.feature_extractor.extract_feature_vector(audio_noisy)
                    all_features.append(feat_noisy)
                    all_labels.append(EMOTION_TO_IDX[emotion])

                    for n_steps in [-2, 2]:
                        audio_pitch = self.audio_loader.augment_pitch_shift(audio, n_steps)
                        feat_pitch = self.feature_extractor.extract_feature_vector(audio_pitch)
                        all_features.append(feat_pitch)
                        all_labels.append(EMOTION_TO_IDX[emotion])

            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"\n  [警告] 处理失败: {file_path} -> {e}")

        features = np.array(all_features)
        labels = np.array(all_labels)

        print(f"\n[完成] 处理结果:")
        print(f"  总样本数: {len(features)}")
        print(f"  特征维度: {features.shape[1] if len(features) > 0 else 0}")
        print(f"  失败样本: {error_count}")

        self._print_class_distribution(labels)

        if save and len(features) > 0:
            self._save_processed_data(features, labels, all_metadata, prefix="casia")

        return features, labels, all_metadata

    def process_directory(
        self,
        directory: Union[str, Path],
        label_func=None,
        save: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        处理指定目录下的所有音频文件

        Args:
            directory: 音频目录
            label_func: 标签提取函数，输入文件名返回情绪标签字符串
            save: 是否保存

        Returns:
            features, labels, filenames
        """
        directory = Path(directory)
        audio_files = self.dataset_manager.scan_directory(directory)
        if not audio_files:
            return np.array([]), np.array([]), []

        all_features = []
        all_labels = []
        all_files = []

        for file_path in tqdm(audio_files, desc="处理音频"):
            try:
                audio = self.audio_loader.preprocess(str(file_path))
                feature = self.feature_extractor.extract_feature_vector(audio)

                if label_func:
                    label_str = label_func(file_path.name)
                    if label_str in EMOTION_TO_IDX:
                        all_labels.append(EMOTION_TO_IDX[label_str])
                    else:
                        continue
                else:
                    all_labels.append(-1)  # 未知标签

                all_features.append(feature)
                all_files.append(str(file_path))

            except Exception as e:
                print(f"[警告] 处理失败: {file_path.name} -> {e}")

        features = np.array(all_features)
        labels = np.array(all_labels)

        if save and len(features) > 0:
            self._save_processed_data(features, labels, all_files, prefix="custom")

        return features, labels, all_files

    def train_test_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        划分训练集、验证集、测试集

        Args:
            features: 特征数组
            labels: 标签数组
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子

        Returns:
            包含训练/验证/测试数据的字典
        """
        from sklearn.model_selection import train_test_split

        np.random.seed(random_seed)

        # 先分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_seed
        )

        # 再从剩余部分分出验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=random_seed
        )

        result = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

        print(f"\n[数据集划分]")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")

        return result

    def save_split_data(self, data_dict: Dict[str, np.ndarray], prefix: str = "ravdess"):
        """
        保存划分后的数据集

        Args:
            data_dict: 数据字典
            prefix: 文件名前缀
        """
        save_dir = PROCESSED_DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, data in data_dict.items():
            filepath = save_dir / f"{prefix}_{name}.npy"
            np.save(str(filepath), data)
            print(f"  [保存] {filepath}")

        # 保存情绪标签映射信息
        info = {
            "num_classes": NUM_EMOTIONS,
            "emotion_labels": EMOTION_LABELS,
            "emotion_to_idx": EMOTION_TO_IDX,
            "feature_dim": data_dict["X_train"].shape[1],
        }
        info_path = save_dir / f"{prefix}_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"  [保存] {info_path}")

    def load_processed_data(self, prefix: str = "ravdess") -> Dict[str, np.ndarray]:
        """
        加载已处理的数据

        Args:
            prefix: 文件名前缀

        Returns:
            数据字典
        """
        save_dir = PROCESSED_DATA_DIR
        data_dict = {}
        names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]

        for name in names:
            filepath = save_dir / f"{prefix}_{name}.npy"
            if filepath.exists():
                data_dict[name] = np.load(str(filepath))
            else:
                print(f"[警告] 文件不存在: {filepath}")
                return {}

        print(f"[OK] 从 {save_dir} 加载数据完成")
        print(f"  训练集: {len(data_dict['X_train'])} 样本")
        print(f"  验证集: {len(data_dict['X_val'])} 样本")
        print(f"  测试集: {len(data_dict['X_test'])} 样本")

        return data_dict

    def _print_class_distribution(self, labels: np.ndarray):
        """打印类别分布"""
        print("\n[类别分布]:")
        for idx in range(NUM_EMOTIONS):
            count = np.sum(labels == idx)
            percentage = count / len(labels) * 100
            print(f"  {IDX_TO_EMOTION[idx]:10s}: {count:4d} ({percentage:5.1f}%)")

    def _save_processed_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        extra_data,
        prefix: str
    ):
        """保存处理后的数据"""
        save_dir = FEATURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存特征和标签
        np.save(save_dir / f"{prefix}_features.npy", features)
        np.save(save_dir / f"{prefix}_labels.npy", labels)

        # 保存元数据
        meta_path = save_dir / f"{prefix}_metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(extra_data, f)

        print(f"\n[数据已保存到] {save_dir}/")
        print(f"  - {prefix}_features.npy: {features.shape}")
        print(f"  - {prefix}_labels.npy: {labels.shape}")
        print(f"  - {prefix}_metadata.pkl")


if __name__ == "__main__":
    # 测试数据预处理
    preprocessor = DataPreprocessor()

    # 处理 RAVDESS 数据集
    features, labels, metadata = preprocessor.process_ravdess(
        data_dir=RAW_DATA_DIR / "RAVDESS",
        save=True,
        augment=False
    )

    if len(features) > 0:
        # 划分数据集
        data_dict = preprocessor.train_test_split(features, labels)
        preprocessor.save_split_data(data_dict, prefix="ravdess")
