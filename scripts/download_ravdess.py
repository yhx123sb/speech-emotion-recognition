"""
RAVDESS 数据集下载脚本
自动下载 24 位演员的音频文件，并解压到 data/raw/RAVDESS 目录
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from speech_emotion_recognition.utils.config import RAW_DATA_DIR


class DownloadProgressBar:
    """下载进度条"""
    def __init__(self, description="下载中"):
        self.pbar = None
        self.description = description

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                desc=self.description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_ravdess(actor_ids=None, dest_dir=None):
    """
    下载 RAVDESS 数据集

    数据来源: https://zenodo.org/record/1188976
    包含 24 位专业演员（12男12女）的 8 种情绪语音

    Args:
        actor_ids: 演员编号列表，默认全部 1-24
        dest_dir: 目标目录
    """
    dest_dir = dest_dir or RAW_DATA_DIR / "RAVDESS"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if actor_ids is None:
        actor_ids = list(range(1, 25))

    base_url = "https://zenodo.org/record/1188976/files"

    print("=" * 60)
    print("下载 RAVDESS 语音情绪数据集")
    print("=" * 60)
    print(f"目标目录: {dest_dir}")
    print(f"演员数量: {len(actor_ids)} (编号: {actor_ids[0]}-{actor_ids[-1]})")
    print(f"预计大小: 约 2.5 GB")
    print("=" * 60)

    total_files = len(actor_ids)
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, actor_id in enumerate(actor_ids, 1):
        filename = f"Actor_{actor_id:02d}.zip"
        file_url = f"{base_url}/{filename}"
        zip_path = dest_dir / filename
        extract_dir = dest_dir / f"Actor_{actor_id:02d}"

        # 如果已解压，跳过
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[{idx}/{total_files}] Actor_{actor_id:02d} 已存在，跳过")
            skip_count += 1
            continue

        try:
            print(f"[{idx}/{total_files}] 下载 Actor_{actor_id:02d} (演员 #{actor_id})...")

            # 下载 zip 文件
            urllib.request.urlretrieve(
                file_url,
                zip_path,
                DownloadProgressBar(f"  Actor_{actor_id:02d}.zip")
            )

            # 解压
            print(f"  解压中...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)

            # 删除 zip 文件，节省空间
            zip_path.unlink()
            success_count += 1
            print(f"  [完成]")

        except urllib.error.HTTPError as e:
            print(f"  [失败] HTTP 错误: {e.code} - {e.reason}")
            fail_count += 1
        except Exception as e:
            print(f"  [失败] {e}")
            fail_count += 1

    # 打印汇总
    print("\n" + "=" * 60)
    print("下载完成汇总")
    print(f"  成功: {success_count} / {total_files}")
    print(f"  跳过: {skip_count} / {total_files}")
    print(f"  失败: {fail_count} / {total_files}")
    print(f"  数据位置: {dest_dir}")
    print("=" * 60)

    return success_count > 0


if __name__ == "__main__":
    # 下载前 2 位演员测试（确认可用后再下载全部）
    # 如果要下载全部，使用 download_ravdess()
    print("请选择下载模式：")
    print("  1. [推荐] 下载前 2 位演员测试（约 200 MB）")
    print("  2. 下载全部 24 位演员（约 2.5 GB）")

    choice = input("请输入 1 或 2: ").strip()

    if choice == "1":
        download_ravdess(actor_ids=[1, 2])
    elif choice == "2":
        confirm = input("确认下载全部 24 位演员？(y/n): ").strip()
        if confirm.lower() == 'y':
            download_ravdess()
        else:
            print("已取消")
    else:
        print("无效输入")
