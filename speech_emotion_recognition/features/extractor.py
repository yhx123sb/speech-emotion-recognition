"""
声学特征提取模块
提取：MFCC、Mel频谱图、色度特征、光谱对比度、过零率、RMS能量等
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple


class FeatureExtractor:
    """
    声学特征提取器
    从音频数据中提取多种用于情绪识别的特征
    """

    # MFCC 参数
    N_MFCC = 40
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS
    ):
        """
        初始化特征提取器

        Args:
            sr: 采样率
            n_mfcc: MFCC 系数数量
            n_fft: FFT 窗口大小
            hop_length: 帧移
            n_mels: Mel 滤波器组数量
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 MFCC 特征 (Mel Frequency Cepstral Coefficients)

        情绪识别中最常用的特征之一，模拟人耳听觉特性

        Args:
            audio: 音频数据

        Returns:
            MFCC 特征矩阵 (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        return mfcc

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 Mel 频谱图

        适合 CNN 输入的二维图像式特征，保留时频信息

        Args:
            audio: 音频数据

        Returns:
            Mel 频谱图 (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # 转换为对数刻度（dB）
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        提取色度特征 (Chroma Feature)

        反映12个音级的能量分布，对音乐性语音（如歌声）的情绪识别有帮助

        Args:
            audio: 音频数据

        Returns:
            色度特征 (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chroma

    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        提取光谱对比度 (Spectral Contrast)

        描述频谱中波峰与波谷的差异，反映音色特征

        Args:
            audio: 音频数据

        Returns:
            光谱对比度 (n_bands+1, time_steps) 默认 7 个频带
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return contrast

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        提取过零率 (Zero Crossing Rate, ZCR)

        衡量音频信号的粗糙度/噪音程度，高过零率常与愤怒等情绪相关

        Args:
            audio: 音频数据

        Returns:
            过零率 (1, time_steps)
        """
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return zcr

    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 RMS 能量 (Root Mean Square Energy)

        反映音频的响度/能量变化，强烈的情绪通常伴随高能量

        Args:
            audio: 音频数据

        Returns:
            RMS 能量 (1, time_steps)
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        return rms

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取所有特征，返回字典

        Args:
            audio: 音频数据

        Returns:
            特征字典，键为特征名，值为特征矩阵
        """
        features = {}
        features['mfcc'] = self.extract_mfcc(audio)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)
        features['chroma'] = self.extract_chroma(audio)
        features['spectral_contrast'] = self.extract_spectral_contrast(audio)
        features['zcr'] = self.extract_zero_crossing_rate(audio)
        features['rms'] = self.extract_rms_energy(audio)
        return features

    def extract_feature_vector(self, audio: np.ndarray) -> np.ndarray:
        """
        提取聚合特征向量（适合传统 ML 模型）

        对每个特征计算统计量：均值、标准差、最大值、最小值

        Args:
            audio: 音频数据

        Returns:
            一维特征向量（长度 = 各特征统计量之和）
        """
        mfcc = self.extract_mfcc(audio)
        chroma = self.extract_chroma(audio)
        contrast = self.extract_spectral_contrast(audio)
        zcr = self.extract_zero_crossing_rate(audio)
        rms = self.extract_rms_energy(audio)

        feature_vector = []

        # MFCC 统计量（每个系数的均值 + 标准差）
        feature_vector.extend(np.mean(mfcc, axis=1))
        feature_vector.extend(np.std(mfcc, axis=1))

        # Chroma 统计量
        feature_vector.extend(np.mean(chroma, axis=1))
        feature_vector.extend(np.std(chroma, axis=1))

        # Spectral Contrast 统计量
        feature_vector.extend(np.mean(contrast, axis=1))
        feature_vector.extend(np.std(contrast, axis=1))

        # ZCR 统计量
        feature_vector.extend(np.mean(zcr, axis=1))
        feature_vector.extend(np.std(zcr, axis=1))

        # RMS 统计量
        feature_vector.extend(np.mean(rms, axis=1))
        feature_vector.extend(np.std(rms, axis=1))

        return np.array(feature_vector)

    def get_feature_vector_dim(self) -> int:
        """
        获取特征向量的维度（用于初始化模型输入层）

        Returns:
            特征向量维度
        """
        dummy_audio = np.zeros(self.sr * 3)  # 3 秒静音
        return len(self.extract_feature_vector(dummy_audio))

    def plot_mel_spectrogram(self, audio: np.ndarray, title: str = "Mel Spectrogram"):
        """
        可视化 Mel 频谱图

        Args:
            audio: 音频数据
            title: 图表标题
        """
        mel_spec_db = self.extract_mel_spectrogram(audio)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            mel_spec_db,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 简单测试
    extractor = FeatureExtractor()
    print(f"特征提取器初始化完成")
    print(f"特征向量维度: {extractor.get_feature_vector_dim()}")

    # 用随机噪声模拟音频测试
    dummy_audio = np.random.randn(16000 * 3)
    features = extractor.extract_all_features(dummy_audio)
    for name, feat in features.items():
        print(f"{name}: shape = {feat.shape}")
