"""
数据预处理执行脚本
处理步骤：
  1. 扫描数据目录
  2. 加载并预处理音频
  3. 提取特征
  4. 划分训练/验证/测试集
  5. 保存处理后的数据

支持的数据集：
  - CASIA 中文情感语料库（本地已有）
  - RAVDESS 英文情感语料库（需下载）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

from speech_emotion_recognition.data.preprocessor import DataPreprocessor
from speech_emotion_recognition.utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
)


def check_casia_available():
    """检查 CASIA 数据是否存在"""
    casia_dir = PROJECT_ROOT / "speech_emotion_recognition" / "casia"
    if not casia_dir.exists():
        print(f"[错误] CASIA 数据目录不存在: {casia_dir}")
        return False
    print(f"[OK] 找到 CASIA 数据目录: {casia_dir}")
    return True


def check_ravdess_available():
    """检查 RAVDESS 数据是否存在"""
    ravdess_dir = RAW_DATA_DIR / "RAVDESS"
    if not ravdess_dir.exists():
        print(f"[提示] RAVDESS 数据目录不存在: {ravdess_dir}")
        print("  可运行: python scripts/download_ravdess.py")
        return False

    actor_dirs = list(ravdess_dir.glob("Actor_*"))
    if not actor_dirs:
        print(f"[提示] RAVDESS 目录中没有找到演员数据: {ravdess_dir}")
        return False

    print(f"[OK] 找到 {len(actor_dirs)} 位演员的 RAVDESS 数据")
    return True


def main():
    print("=" * 60)
    print("数据预处理 Pipeline")
    print("=" * 60)
    print("\n选择要处理的数据集：")
    print("  1. CASIA 中文情感语料库（本地已有，推荐！）")
    print("  2. RAVDESS 英文情感语料库（需先下载）")

    dataset_choice = input("\n请输入 1 或 2: ").strip()

    if dataset_choice == "1":
        if not check_casia_available():
            fallback = input("是否尝试使用 RAVDESS？(y/n): ").strip()
            if fallback.lower() == 'y':
                dataset_choice = "2"
            else:
                return
        else:
            prefix = "casia"
    if dataset_choice == "2":
        if not check_ravdess_available():
            print("[错误] RAVDESS 数据也不可用，请先下载。")
            return
        prefix = "ravdess"

    print("\n请选择数据增强模式：")
    print("  1. [推荐] 不使用增强（快速测试）")
    print("  2. 使用数据增强（更多样本，效果更好）")

    choice = input("请输入 1 或 2: ").strip()
    use_augment = (choice == "2")

    # 初始化预处理器
    preprocessor = DataPreprocessor()

    # 处理数据集
    if prefix == "casia":
        features, labels, metadata = preprocessor.process_casia(
            save=True,
            augment=use_augment
        )
    else:
        features, labels, metadata = preprocessor.process_ravdess(
            data_dir=RAW_DATA_DIR / "RAVDESS",
            save=True,
            augment=use_augment
        )

    if len(features) == 0:
        print("[错误] 数据处理失败，没有生成有效样本")
        return
    # 划分数据集
    print("\n" + "=" * 60)
    print("划分训练/验证/测试集")
    print("=" * 60)

    data_dict = preprocessor.train_test_split(
        features,
        labels,
        test_size=0.2,
        val_size=0.1,
        random_seed=42
    )

    # 保存划分后的数据
    preprocessor.save_split_data(data_dict, prefix=prefix)

    print("\n" + "=" * 60)
    print("[完成] 数据处理全部结束！")
    print(f"特征维度: {features.shape[1]}")
    print(f"情绪类别: {len(set(labels))} 种")
    print(f"数据已保存到: {PROCESSED_DATA_DIR / f'{prefix}_*.npy'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
