"""
项目全局配置
定义常量、路径、参数等
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


# ==================== 项目路径 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# ==================== 音频参数 ====================
TARGET_SR = 16000  # 目标采样率 (Hz)
TARGET_DURATION = 3.0  # 音频切片时长 (秒)
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)  # 目标样本数

# ==================== 特征参数 ====================
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# ==================== 情绪标签 ====================
# RAVDESS 数据集情绪标签映射
EMOTION_LABELS_RAVDESS = {
    "01": "neutral",      # 中性
    "02": "calm",         # 平静
    "03": "happy",        # 开心
    "04": "sad",          # 悲伤
    "05": "angry",        # 愤怒
    "06": "fearful",      # 恐惧
    "07": "disgust",      # 厌恶
    "08": "surprised",    # 惊讶
}

# 通用情绪类别（简化版）
EMOTION_LABELS = [
    "neutral",   # 中性
    "happy",     # 开心
    "sad",       # 悲伤
    "angry",     # 愤怒
    "fearful",   # 恐惧
    "disgust",   # 厌恶
    "surprised", # 惊讶
]

NUM_EMOTIONS = len(EMOTION_LABELS)

# 情绪标签到索引的映射
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}


# ==================== 训练参数 ====================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据参数
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    batch_size: int = 32

    # 传统机器学习参数
    svm_kernel: str = "rbf"
    svm_c: float = 1.0

    # 深度学习参数
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 10  # 早停耐心值
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    # 模型保存
    model_name: str = "emotion_recognition_model"


# ==================== 数据增强参数 ====================
@dataclass
class AugmentationConfig:
    """数据增强配置"""
    noise_level: float = 0.005  # 高斯噪声强度
    pitch_shift_range: Tuple[int, int] = (-2, 2)  # 音调变化范围（半音）
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)  # 时间拉伸范围
    augment_factor: int = 2  # 增强倍数


# 全局实例
training_cfg = TrainingConfig()
augmentation_cfg = AugmentationConfig()


def ensure_dirs():
    """确保所有数据目录存在"""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, SAVED_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("✅ 所有目录已就绪")


if __name__ == "__main__":
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"情绪类别数: {NUM_EMOTIONS}")
    print(f"情绪标签: {EMOTION_LABELS}")
    print(f"特征向量维度参考（传统 ML）: ...")
    ensure_dirs()
