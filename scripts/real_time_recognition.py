"""
实时语音情绪识别（麦克风录制 + 模型推理）
使用预训练模型实时识别麦克风输入的语音情绪

支持：
- 录制3秒音频 → 提取特征 → 模型推理 → 显示结果
- 支持 SVM、Random Forest、MLP 三种模型
- 连续识别模式
- GPU 加速推理
"""


import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("[错误] 请安装 sounddevice 和 soundfile: pip install sounddevice soundfile")
    sys.exit(1)

from speech_emotion_recognition.features.extractor import FeatureExtractor
from speech_emotion_recognition.utils.config import (
    SAVED_MODELS_DIR,
    EMOTION_LABELS,
    TARGET_SR,
)

# CASIA 数据集的 6 个情绪标签（无 disgust）
CASIA_EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]


class RealtimeEmotionRecognizer:
    """
    实时语音情绪识别器
    从麦克风录制音频 → 提取特征 → 模型推理
    """

    def __init__(self, model_type: str = "mlp"):
        """
        初始化识别器

        Args:
            model_type: 模型类型 ('svm', 'rf', 'mlp')
        """
        self.sr = TARGET_SR
        self.duration = 3.0  # 录制时长（秒）
        self.feature_extractor = FeatureExtractor(sr=self.sr)

        # 加载模型
        self.model = None
        self.model_type = model_type
        self._load_model(model_type)

        # 情绪标签（使用 CASIA 的6个类别）
        self.emotion_labels = CASIA_EMOTION_LABELS
        self.n_classes = len(self.emotion_labels)

        print(f"[初始化] 实时语音情绪识别器")
        print(f"[初始化] 模型: {model_type.upper()}")
        print(f"[初始化] 采样率: {self.sr} Hz")
        print(f"[初始化] 录制时长: {self.duration} 秒")
        print(f"[初始化] 情绪类别: {self.emotion_labels}")

    def _load_model(self, model_type: str):
        """加载预训练模型"""
        if model_type == "mlp":
            model_path = SAVED_MODELS_DIR / "mlp_model.pt"
            if not model_path.exists():
                print(f"[错误] MLP 模型不存在: {model_path}")
                print("[提示] 请先运行训练脚本训练 MLP 模型")
                sys.exit(1)

            import torch
            from speech_emotion_recognition.models.cnn_model import EmotionMLP

            # 加载 checkpoint
            checkpoint = torch.load(str(model_path), map_location="cpu")

            # 获取类别数和隐藏层维度
            if "num_classes" in checkpoint:
                n_classes = checkpoint["num_classes"]
            else:
                n_classes = 6

            # 从 checkpoint 推断隐藏层维度
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 从权重形状推断隐藏层维度
            hidden_dims = [256, 128, 64]  # 默认
            if "network.0.weight" in state_dict:
                shape0 = state_dict["network.0.weight"].shape  # (hidden1, input_dim)
                shape4 = state_dict["network.4.weight"].shape  # (hidden2, hidden1)
                shape8 = state_dict["network.8.weight"].shape  # (hidden3, hidden2)
                hidden_dims = [shape0[0], shape4[0], shape8[0]]

            # 创建模型并加载权重
            self.model = EmotionMLP(
                input_dim=122,
                n_classes=n_classes,
                hidden_dims=hidden_dims,
            )
            self.model.load_state_dict(state_dict)

            self.model.eval()

            # 尝试使用 GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"[加载] MLP 模型已加载 (设备: {self.device})")

        elif model_type == "svm":
            model_path = SAVED_MODELS_DIR / "svm_model.pkl"
            if not model_path.exists():
                print(f"[错误] SVM 模型不存在: {model_path}")
                sys.exit(1)

            import joblib
            self.model = joblib.load(str(model_path))
            self.device = "cpu"
            print(f"[加载] SVM 模型已加载")

        elif model_type == "rf":
            model_path = SAVED_MODELS_DIR / "random_forest_model.pkl"
            if not model_path.exists():
                print(f"[错误] Random Forest 模型不存在: {model_path}")
                sys.exit(1)

            import joblib
            self.model = joblib.load(str(model_path))
            self.device = "cpu"
            print(f"[加载] Random Forest 模型已加载")

        else:
            print(f"[错误] 不支持的模型类型: {model_type}")
            print("[提示] 可选: 'svm', 'rf', 'mlp'")
            sys.exit(1)

    def list_audio_devices(self):
        """列出可用的音频输入设备"""
        print("\n" + "=" * 60)
        print("  可用音频输入设备")
        print("=" * 60)

        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append((i, device))
                print(f"  [{i}] {device['name'].encode('ascii', errors='replace').decode('ascii')}")
                print(f"      输入通道: {device['max_input_channels']}")
                print(f"      默认采样率: {device['default_samplerate']}")
                print()

        return input_devices

    def record_audio(self, duration: float = None, device_id: int = None) -> np.ndarray:
        """
        从麦克风录制音频

        Args:
            duration: 录制时长（秒），默认使用初始化时的值
            device_id: 音频设备ID，None 使用默认设备

        Returns:
            录制的音频数据 (numpy array)
        """
        if duration is None:
            duration = self.duration

        print(f"\n[录音] 正在录制 {duration} 秒...", end="", flush=True)

        # 倒计时
        for i in range(int(duration), 0, -1):
            print(f" {i}", end="", flush=True)
            time.sleep(1)
        print(" 开始!")

        # 录制音频
        recording = sd.rec(
            int(duration * self.sr),
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=device_id,
        )
        sd.wait()

        # 转换为1维数组
        audio = recording.flatten()

        print(f"[录音] 完成! 音频长度: {len(audio)} 样本 ({len(audio)/self.sr:.2f}s)")

        return audio

    def predict_emotion(self, audio: np.ndarray) -> dict:
        """
        预测音频的情感

        Args:
            audio: 音频数据

        Returns:
            预测结果字典
        """
        # 提取特征
        feature_vector = self.feature_extractor.extract_feature_vector(audio)

        if self.model_type == "mlp":
            import torch

            # 转换为 tensor
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # 获取预测
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

        else:
            # sklearn 模型（SVM/RF）
            feature_vector = feature_vector.reshape(1, -1)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feature_vector)[0]
            else:
                probs = None

            predicted_class = int(self.model.predict(feature_vector)[0])

            if probs is not None:
                confidence = float(probs[predicted_class])
            else:
                confidence = 1.0

        # 构建结果
        emotion_name = self.emotion_labels[predicted_class] if predicted_class < len(self.emotion_labels) else f"未知({predicted_class})"

        result = {
            "emotion_index": predicted_class,
            "emotion": emotion_name,
            "confidence": confidence,
            "probabilities": {},
        }

        # 所有类别的概率
        for i, label in enumerate(self.emotion_labels):
            if self.model_type == "mlp" and "probs" in dir():
                prob = probs[i] if i < len(probs) else 0.0
            else:
                prob = probs[i] if probs is not None and i < len(probs) else 0.0
            result["probabilities"][label] = prob

        return result

    def display_result(self, result: dict):
        """
        显示预测结果

        Args:
            result: 预测结果字典
        """
        emotion = result["emotion"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        # 获取情绪对应的表情符号
        emoji_map = {
            "neutral": "😐",
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fearful": "😨",
            "surprised": "😲",
        }
        emoji = emoji_map.get(emotion, "🤔")

        print("\n" + "=" * 60)
        print(f"  识别结果")
        print("=" * 60)
        print(f"  情绪: {emoji} {emotion.upper()}")
        print(f"  置信度: {confidence:.2%}")
        print(f"  ─────────────────────────────────────────")
        print(f"  详细概率:")
        print()

        # 按概率排序显示
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            emoji_l = emoji_map.get(label, "")
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"    {emoji_l} {label:>10}: {bar} {prob:.1%}")

        print("=" * 60)

    def run_once(self, device_id: int = None):
        """
        运行一次实时识别

        Args:
            device_id: 音频设备ID
        """
        # 录制音频
        audio = self.record_audio(device_id=device_id)

        # 预测
        result = self.predict_emotion(audio)

        # 显示结果
        self.display_result(result)

        return result

    def run_continuous(self, device_id: int = None, interval: float = 1.0):
        """
        连续识别模式

        Args:
            device_id: 音频设备ID
            interval: 识别间隔（秒），录制之间等待时间
        """
        print("\n" + "=" * 60)
        print("  连续识别模式")
        print("  按 Ctrl+C 停止")
        print("=" * 60)

        try:
            round_count = 0
            smooth_probs = None  # 用于平滑预测

            while True:
                round_count += 1
                print(f"\n--- 第 {round_count} 轮识别 ---")

                # 录制
                audio = self.record_audio(device_id=device_id)

                # 预测
                result = self.predict_emotion(audio)

                # 平滑处理（移动平均）
                if smooth_probs is None:
                    smooth_probs = result["probabilities"].copy()
                else:
                    alpha = 0.7  # 新数据的权重
                    for key in smooth_probs:
                        smooth_probs[key] = (
                            alpha * result["probabilities"][key]
                            + (1 - alpha) * smooth_probs[key]
                        )

                # 使用平滑后的结果
                smooth_result = result.copy()
                smooth_result["probabilities"] = smooth_probs
                smooth_result["emotion"] = max(smooth_probs, key=smooth_probs.get)
                smooth_result["confidence"] = smooth_probs[smooth_result["emotion"]]
                smooth_result["emotion_index"] = self.emotion_labels.index(
                    smooth_result["emotion"]
                )

                self.display_result(smooth_result)

                # 等待间隔
                if interval > 0:
                    print(f"\n  等待 {interval} 秒后开始下一轮...")
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[停止] 连续识别已停止")
            print(f"  总共识别: {round_count} 轮")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="实时语音情绪识别")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["svm", "rf", "mlp"],
        help="选择模型 (默认: mlp)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "continuous"],
        help="识别模式: single=单次, continuous=连续 (默认: single)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出可用的音频输入设备",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="音频输入设备ID (默认: 系统默认)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="连续模式下的识别间隔（秒）",
    )

    args = parser.parse_args()

    # 创建识别器
    recognizer = RealtimeEmotionRecognizer(model_type=args.model)

    # 列出设备
    if args.list_devices:
        recognizer.list_audio_devices()
        return

    # 运行识别
    if args.mode == "continuous":
        recognizer.run_continuous(device_id=args.device, interval=args.interval)
    else:
        input("\n按 Enter 开始录音...")
        recognizer.run_once(device_id=args.device)


if __name__ == "__main__":
    main()
