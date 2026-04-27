"""
语音情绪识别系统 - 主入口
项目: Speech Emotion Recognition (SER)
"""

import sys
import io

# 解决 Windows 终端 emoji 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from speech_emotion_recognition.utils.config import ensure_dirs
from speech_emotion_recognition.data.audio_loader import AudioLoader
from speech_emotion_recognition.features.extractor import FeatureExtractor


def main():
    print("=" * 60)
    print("语音情绪识别系统 (Speech Emotion Recognition)")
    print("=" * 60)

    # 确保目录结构存在
    ensure_dirs()

    # 初始化核心模块
    audio_loader = AudioLoader()
    feature_extractor = FeatureExtractor()

    print("\n[OK] 第 1 阶段完成！")
    print(f"   - 音频加载器: 采样率={audio_loader.target_sr}Hz")
    print(f"   - 特征提取器: MFCC={feature_extractor.n_mfcc}, Mel波段={feature_extractor.n_mels}")
    print(f"   - 特征向量维度(传统ML模型输入): {feature_extractor.get_feature_vector_dim()}")
    print("\n[下一步] 下载数据集 -> 第 2 阶段")
    print("=" * 60)


if __name__ == "__main__":
    main()
