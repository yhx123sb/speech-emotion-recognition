"""
音频加载与预处理模块
负责：加载各种格式的音频文件、统一采样率、音量归一化、静音检测与去除
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class AudioLoader:
    """音频加载器，统一处理不同格式的音频文件"""

    TARGET_SR = 16000  # 目标采样率 16kHz
    TARGET_DURATION = 3.0  # 目标音频时长 3 秒

    def __init__(self, target_sr: int = TARGET_SR, target_duration: float = TARGET_DURATION):
        """
        初始化音频加载器

        Args:
            target_sr: 目标采样率，默认 16000 Hz
            target_duration: 目标音频时长（秒），默认 3 秒
        """
        self.target_sr = target_sr
        self.target_length = int(target_sr * target_duration)

    def load(self, file_path: Union[str, Path], sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            file_path: 音频文件路径
            sr: 采样率，若为 None 则使用 target_sr

        Returns:
            audio: 音频数据 (numpy array)
            sr: 实际采样率
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {file_path}")

        sr = sr or self.target_sr
        audio, orig_sr = librosa.load(str(file_path), sr=sr, mono=True)
        return audio, sr

    def normalize_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        音量归一化（峰值归一化到 [-1, 1]）

        Args:
            audio: 原始音频数据

        Returns:
            归一化后的音频数据
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def remove_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        去除音频首尾的静音部分

        Args:
            audio: 音频数据
            top_db: 静音阈值（dB），低于此值视为静音
            frame_length: 帧长度
            hop_length: 帧移

        Returns:
            去除静音后的音频数据
        """
        trimmed_audio, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed_audio

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        将音频填充或裁剪到固定长度

        Args:
            audio: 音频数据

        Returns:
            固定长度的音频数据
        """
        if len(audio) > self.target_length:
            # 裁剪到目标长度
            return audio[:self.target_length]
        elif len(audio) < self.target_length:
            # 用零填充到目标长度
            padding = self.target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        else:
            return audio

    def preprocess(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        完整的音频预处理 pipeline：
        加载 → 归一化 → 去静音 → 统一长度

        Args:
            file_path: 音频文件路径

        Returns:
            预处理后的音频数据
        """
        audio, sr = self.load(file_path)
        audio = self.normalize_volume(audio)
        audio = self.remove_silence(audio)
        audio = self.pad_or_truncate(audio)
        return audio

    def augment_add_noise(
        self,
        audio: np.ndarray,
        noise_level: float = 0.005
    ) -> np.ndarray:
        """
        数据增强：添加高斯白噪声

        Args:
            audio: 音频数据
            noise_level: 噪声强度

        Returns:
            添加噪声后的音频
        """
        noise = np.random.randn(len(audio)) * noise_level
        return audio + noise

    def augment_pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: int = 2
    ) -> np.ndarray:
        """
        数据增强：音调变换

        Args:
            audio: 音频数据
            n_steps: 半音数（正数升高，负数降低）

        Returns:
            音调变换后的音频
        """
        return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)

    def augment_time_stretch(
        self,
        audio: np.ndarray,
        rate: float = 1.2
    ) -> np.ndarray:
        """
        数据增强：时间拉伸

        Args:
            audio: 音频数据
            rate: 拉伸率（>1 加快，<1 减慢）

        Returns:
            拉伸后的音频
        """
        return librosa.effects.time_stretch(audio, rate=rate)


if __name__ == "__main__":
    # 简单测试
    loader = AudioLoader()
    print(f"音频加载器初始化完成")
    print(f"目标采样率: {loader.target_sr} Hz")
    print(f"目标时长: {loader.TARGET_DURATION} 秒")
    print(f"目标样本数: {loader.target_length}")
