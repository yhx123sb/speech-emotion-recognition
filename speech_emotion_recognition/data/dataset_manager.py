"""
数据集管理与下载模块
支持 RAVDESS、TESS、CREMA-D、Emo-DB 等常见数据集的下载与组织
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from ..utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
import os as os_module


class DatasetManager:
    """
    数据集管理器
    负责下载、解压、组织常见语音情绪数据集
    """

    # CASIA 情绪标签映射（目录名 -> 标准标签）
    CASIA_EMOTION_MAP = {
        "angry": "angry",
        "fear": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprised",
    }

    # 公开数据集信息
    DATASETS = {
        "ravdess": {
            "name": "RAVDESS",
            "url": "https://zenodo.org/record/1188976/files/Actor_%02d.zip",
            "description": "瑞尔森音频-视觉情感语音与歌曲数据集"
        },
        "tess": {
            "name": "TESS",
            "url": "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess",
            "description": "多伦多情绪语音集"
        },
        "crema_d": {
            "name": "CREMA-D",
            "url": "https://www.kaggle.com/datasets/ejlok1/cremad",
            "description": "视听情感分析数据集"
        },
        "emo_db": {
            "name": "Emo-DB",
            "url": "https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb",
            "description": "柏林情绪语音数据库"
        }
    }

    # RAVDESS 文件名规则解析
    # 格式: 03-01-01-01-01-01-01.wav
    # 03 = 模态 (01=全视频, 02=仅视频, 03=仅音频)
    # 01 = 语音类型 (01=语音, 02=歌曲)
    # 01 = 情绪 (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
    # 01 = 情绪强度 (01=normal, 02=strong)
    # 01 = 语句 (01="Kids are talking by the door", 02="Dogs are sitting by the door")
    # 01 = 重复次数 (01=1st repetition, 02=2nd repetition)
    # 01 = 演员编号 (01-24 专业演员, 奇数=男, 偶数=女)

    RAVDESS_EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    @staticmethod
    def parse_ravdess_filename(filename: str) -> Optional[dict]:
        """
        解析 RAVDESS 文件名，提取元数据

        Args:
            filename: 文件名 (如 03-01-01-01-01-01-01.wav)

        Returns:
            元数据字典，解析失败返回 None
        """
        try:
            parts = filename.replace(".wav", "").split("-")
            if len(parts) < 7:
                return None

            emotion_code = parts[2]
            emotion = DatasetManager.RAVDESS_EMOTION_MAP.get(emotion_code, "unknown")

            return {
                "modality": parts[0],
                "vocal_type": parts[1],
                "emotion": emotion,
                "emotion_code": emotion_code,
                "intensity": "strong" if parts[3] == "02" else "normal",
                "statement": parts[4],
                "repetition": parts[5],
                "actor": parts[6],
                "gender": "male" if int(parts[6]) % 2 == 1 else "female"
            }
        except Exception:
            return None

    def download_ravdess(self, actors: List[int] = None, dest_dir: Path = None):
        """
        下载 RAVDESS 数据集
        注意：需要下载完整 zip，这里给出下载指引

        Args:
            actors: 演员编号列表，默认为 None（下载全部）
            dest_dir: 目标目录
        """
        dest_dir = dest_dir or RAW_DATA_DIR / "RAVDESS"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if actors is None:
            actors = list(range(1, 25))  # 24 位演员

        base_url = "https://zenodo.org/record/1188976/files"

        print("=" * 60)
        print("🌐 RAVDESS 数据集下载指引")
        print("=" * 60)
        print(f"\n由于文件较大，请手动下载：")
        print(f"下载地址: https://zenodo.org/record/1188976")
        print(f"\n下载后解压到: {dest_dir}")
        print(f"\n或者使用以下命令（需要 wget）：")
        for actor_id in actors:
            url = f"{base_url}/Actor_{actor_id:02d}.zip"
            print(f"wget {url} -P {dest_dir}")
        print("=" * 60)

    def organize_ravdess(self, source_dir: Path = None) -> List[Tuple[str, str, str]]:
        """
        组织和解析 RAVDESS 数据集

        Args:
            source_dir: RAVDESS 数据目录

        Returns:
            (文件路径, 情绪标签, 元数据) 列表
        """
        source_dir = source_dir or RAW_DATA_DIR / "RAVDESS"
        if not source_dir.exists():
            print(f"❌ 目录不存在: {source_dir}")
            return []

        samples = []
        for actor_dir in source_dir.iterdir():
            if not actor_dir.is_dir():
                continue
            for wav_file in actor_dir.glob("*.wav"):
                meta = self.parse_ravdess_filename(wav_file.name)
                if meta:
                    samples.append((str(wav_file), meta["emotion"], meta))
                else:
                    print(f"⚠️ 无法解析文件名: {wav_file.name}")

        print(f"✅ 找到 {len(samples)} 个 RAVDESS 音频样本")
        return samples

    def organize_casia(self, data_dir: Path = None) -> List[Tuple[str, str, dict]]:
        """
        组织和解析 CASIA 中文情感语料库

        CASIA 目录结构:
        casia/
          ActorName/
            emotion/
              xxx.wav

        Args:
            data_dir: CASIA 数据目录

        Returns:
            (文件路径, 情绪标签, 元数据) 列表
        """
        if data_dir is None:
            # 自动查找可能的 CASIA 路径
            candidates = [
                PROJECT_ROOT / "speech_emotion_recognition" / "casia",
                RAW_DATA_DIR / "casia",
            ]
            for c in candidates:
                if c.exists():
                    data_dir = c
                    break

        if data_dir is None or not data_dir.exists():
            print(f"[错误] CASIA 数据目录不存在: {data_dir}")
            return []

        samples = []

        # 用 os.walk 避免中文路径问题
        for root, dirs, files in os_module.walk(str(data_dir)):
            for f in files:
                if not f.endswith('.wav'):
                    continue

                filepath = os.path.join(root, f)
                rel_path = os_module.path.relpath(filepath, str(data_dir))
                parts = rel_path.split(os_module.sep)

                if len(parts) >= 2:
                    actor_name = parts[0]
                    emotion_dir = parts[1]

                    if emotion_dir in self.CASIA_EMOTION_MAP:
                        emotion = self.CASIA_EMOTION_MAP[emotion_dir]
                        meta = {
                            "actor": actor_name,
                            "emotion_dir": emotion_dir,
                            "filename": f,
                        }
                        samples.append((filepath, emotion, meta))

        print(f"[OK] 找到 {len(samples)} 个 CASIA 音频样本")
        if samples:
            actors = set(s[2]["actor"] for s in samples)
            emotions = set(s[1] for s in samples)
            print(f"  演员: {', '.join(sorted(actors))}")
            print(f"  情绪: {', '.join(sorted(emotions))}")

        return samples

    def scan_directory(self, directory: Path) -> List[Path]:
        """
        扫描目录下所有音频文件

        Args:
            directory: 目录路径

        Returns:
            音频文件路径列表
        """
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = []
        directory = Path(directory)

        if not directory.exists():
            print(f"❌ 目录不存在: {directory}")
            return []

        for ext in audio_extensions:
            audio_files.extend(directory.rglob(f"*{ext}"))

        print(f"✅ 在 {directory} 中找到 {len(audio_files)} 个音频文件")
        return sorted(audio_files)


if __name__ == "__main__":
    manager = DatasetManager()

    # 测试 RAVDESS 文件名解析
    test_name = "03-01-01-01-01-01-01.wav"
    meta = manager.parse_ravdess_filename(test_name)
    print(f"文件名解析测试: {test_name}")
    print(f"元数据: {meta}")

    # 显示下载指引
    manager.download_ravdess()
