"""
数据探索与可视化脚本
用于查看数据集概况、绘制样本分布、频谱图等
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from speech_emotion_recognition.data.audio_loader import AudioLoader
from speech_emotion_recognition.features.extractor import FeatureExtractor
from speech_emotion_recognition.utils.config import (
    RAW_DATA_DIR, EMOTION_LABELS, IDX_TO_EMOTION, RAW_DATA_DIR, FEATURES_DIR
)


def explore_dataset_overview():
    """查看数据集概况"""
    print("=" * 60)
    print("数据集概况")
    print("=" * 60)

    ravdess_dir = RAW_DATA_DIR / "RAVDESS"
    if not ravdess_dir.exists():
        print("[提示] 数据集尚未下载")
        return

    actor_dirs = sorted(ravdess_dir.glob("Actor_*"))
    total_files = 0
    actor_info = []

    for actor_dir in actor_dirs:
        wav_files = list(actor_dir.glob("*.wav"))
        total_files += len(wav_files)
        # 从文件夹名获取演员编号
        actor_num = int(actor_dir.name.split("_")[1])
        gender = "男" if actor_num % 2 == 1 else "女"
        actor_info.append((actor_dir.name, gender, len(wav_files)))

    print(f"演员数量: {len(actor_dirs)} 位")
    print(f"音频总数: {total_files} 个")
    print(f"\n演员列表:")
    for name, gender, count in actor_info:
        print(f"  {name} ({gender}): {count} 个音频")


def plot_sample_distribution():
    """绘制各类情绪样本分布图"""
    labels_path = FEATURES_DIR / "ravdess_labels.npy"
    if not labels_path.exists():
        print("[提示] 请先运行数据预处理")
        return

    labels = np.load(labels_path)
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [IDX_TO_EMOTION[i] for i in unique],
        counts,
        color=['#808080', '#4CAF50', '#2196F3', '#F44336', '#9C27B0', '#FF9800', '#00BCD4']
    )
    plt.title("RAVDESS 数据集各类情绪样本分布", fontsize=14)
    plt.xlabel("情绪类别", fontsize=12)
    plt.ylabel("样本数量", fontsize=12)
    plt.xticks(rotation=30)

    # 在柱子上显示数值
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("outputs/emotion_distribution.png", dpi=150)
    plt.show()
    print("[OK] 样本分布图已保存到 outputs/emotion_distribution.png")


def visualize_audio_waveform(audio_path: str = None):
    """可视化音频波形和频谱图"""
    loader = AudioLoader()
    extractor = FeatureExtractor()

    if audio_path is None:
        # 找一个示例文件
        ravdess_dir = RAW_DATA_DIR / "RAVDESS"
        if not ravdess_dir.exists():
            print("[提示] 数据集不存在，无法展示示例")
            return
        actor_dirs = list(ravdess_dir.glob("Actor_*"))
        if not actor_dirs:
            return
        wav_files = list(actor_dirs[0].glob("*.wav"))
        if not wav_files:
            return
        audio_path = str(wav_files[0])

    print(f"可视化音频: {audio_path}")

    # 加载音频
    audio, sr = loader.load(audio_path)
    audio = loader.normalize_volume(audio)

    # 提取 Mel 频谱图
    mel_spec = extractor.extract_mel_spectrogram(audio)

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # 波形图
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time, audio, color='b', linewidth=0.5)
    axes[0].set_title("音频波形", fontsize=12)
    axes[0].set_xlabel("时间 (秒)", fontsize=10)
    axes[0].set_ylabel("振幅", fontsize=10)
    axes[0].set_xlim([0, len(audio)/sr])

    # Mel 频谱图
    import librosa.display
    img = librosa.display.specshow(
        mel_spec, sr=sr, hop_length=extractor.hop_length,
        x_axis='time', y_axis='mel', ax=axes[1]
    )
    axes[1].set_title("Mel 频谱图", fontsize=12)
    axes[1].set_xlabel("时间 (秒)", fontsize=10)
    axes[1].set_ylabel("频率 (Mel)", fontsize=10)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig("outputs/audio_visualization.png", dpi=150)
    plt.show()
    print("[OK] 波形图已保存到 outputs/audio_visualization.png")


def main():
    print("数据探索与可视化")
    print("=" * 60)

    explore_dataset_overview()

    print("\n" + "=" * 60)
    plot_sample_distribution()

    print("\n" + "=" * 60)
    visualize_audio_waveform()

    print("\n[完成] 探索结束")


if __name__ == "__main__":
    # 确保输出目录存在
    Path("outputs").mkdir(exist_ok=True)
    main()
