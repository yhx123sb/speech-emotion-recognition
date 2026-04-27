"""
训练脚本
训练和评估所有模型（SVM, Random Forest, MLP, CNN, ResNet）
支持加载预处理后的数据、训练、评估、保存模型和结果
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_emotion_recognition.utils.config import (
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    EMOTION_LABELS,
    training_cfg,
)
from speech_emotion_recognition.models.traditional_ml import (
    SVMClassifier,
    RandomForestClassifier,
    train_and_evaluate_traditional_models,
)
from speech_emotion_recognition.models.cnn_model import (
    EmotionCNN,
    EmotionResNet,
    EmotionMLP,
    create_cnn_model,
)
from speech_emotion_recognition.models.trainer import (
    Trainer,
    train_and_evaluate_deep_model,
)
from speech_emotion_recognition.evaluation.metrics import (
    MetricsEvaluator,
    evaluate_model_comprehensive,
    compare_and_report,
)


def load_casia_data() -> dict:
    """
    加载预处理后的 CASIA 数据集

    Returns:
        数据字典，包含 X_train, y_train, X_val, y_val, X_test, y_test
    """
    data_dir = PROCESSED_DATA_DIR
    prefix = "casia"

    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    data_dict = {}

    print(f"[加载] 从 {data_dir} 加载 CASIA 数据...")
    for name in names:
        filepath = data_dir / f"{prefix}_{name}.npy"
        if filepath.exists():
            data_dict[name] = np.load(str(filepath))
            print(f"  [OK] {name}: {data_dict[name].shape}")
        else:
            print(f"[错误] 文件不存在: {filepath}")
            return {}

    print(f"[完成] 数据加载成功!")
    print(f"  训练集: {len(data_dict['X_train'])} 样本")
    print(f"  验证集: {len(data_dict['X_val'])} 样本")
    print(f"  测试集: {len(data_dict['X_test'])} 样本")

    # 连续化标签（处理跳过的索引，如 CASIA 中没有 disgust 导致标签为 0,1,2,3,4,6）
    all_y = np.concatenate([data_dict["y_train"], data_dict["y_val"], data_dict["y_test"]])
    unique_labels = np.sort(np.unique(all_y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    print(f"[信息] 标签连续化映射: {label_map}")
    for key in ["y_train", "y_val", "y_test"]:
        data_dict[key] = np.array([label_map[y] for y in data_dict[key]])

    return data_dict


def train_traditional_models(data_dict: dict) -> dict:
    """
    训练传统 ML 模型 (SVM, Random Forest)

    Args:
        data_dict: 数据字典

    Returns:
        模型结果字典
    """
    print("\n" + "=" * 60)
    print("  1. 训练传统机器学习模型")
    print("=" * 60)

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    results = train_and_evaluate_traditional_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        use_grid_search=False,  # 设为 True 可启用网格搜索（较慢）
    )

    return results


def train_mlp_model(data_dict: dict) -> dict:
    """
    训练 MLP 深度学习模型（基于122维特征向量）

    Args:
        data_dict: 数据字典

    Returns:
        模型结果字典
    """
    print("\n" + "=" * 60)
    print("  2. 训练 MLP 深度模型")
    print("=" * 60)

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"[信息] 输入维度: {input_dim}, 类别数: {n_classes}")

    model = EmotionMLP(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
    )

    trainer, result = train_and_evaluate_deep_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=training_cfg.batch_size,
        num_epochs=training_cfg.num_epochs,
        learning_rate=training_cfg.learning_rate,
        model_name="mlp_model",
    )

    return {"trainer": trainer, **result}


def train_cnn_model(data_dict: dict) -> dict:
    """
    训练 CNN 模型
    注意：需要 Mel 频谱图特征，需要额外从音频文件提取

    Args:
        data_dict: 数据字典

    Returns:
        模型结果字典
    """
    print("\n" + "=" * 60)
    print("  3. 训练 CNN 模型 (基于原始音频)")
    print("=" * 60)
    print("[提示] CNN 需要 Mel 频谱图输入，正在准备数据...")

    # 加载元数据获取文件路径
    import pickle
    from speech_emotion_recognition.data.audio_loader import AudioLoader
    from speech_emotion_recognition.features.extractor import FeatureExtractor
    from speech_emotion_recognition.utils.config import TARGET_SR, FEATURES_DIR

    # 加载元数据
    meta_path = FEATURES_DIR / "casia_metadata.pkl"
    if not meta_path.exists():
        print("[错误] 找不到元数据文件，无法重建 Mel 频谱图")
        print("[建议] 使用 MLP 模型代替，或先运行特征提取")
        return {}

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"[信息] 从元数据重建 Mel 频谱图特征...")

    audio_loader = AudioLoader()
    extractor = FeatureExtractor(sr=TARGET_SR)

    # 提取训练/验证/测试集的 Mel 频谱图
    # 注意：这里简单处理，用所有数据的索引来划分
    # 实际更严谨的做法是在预处理时就保存 Mel 频谱图
    all_features = []
    error_count = 0

    for i, meta in enumerate(metadata):
        try:
            audio = audio_loader.preprocess(meta["file"])
            mel_spec = extractor.extract_mel_spectrogram(audio)
            all_features.append(mel_spec)
        except Exception as e:
            error_count += 1
            all_features.append(np.zeros((128, 94)))  # 用零填充失败样本

        if (i + 1) % 200 == 0:
            print(f"  处理进度: {i+1}/{len(metadata)}")

    all_mel_features = np.array(all_features)
    print(f"[完成] Mel 频谱图提取完成: {all_mel_features.shape}")
    if error_count > 0:
        print(f"[警告] {error_count} 个样本提取失败，已用零填充")

    # 使用数据集的标签
    all_labels = data_dict.get("y_all", np.load(str(PROCESSED_DATA_DIR / "casia_y_train.npy")))

    # 需要重新划分数据集
    # 简化：用原来划分好的索引重新提取
    # 更严谨：在预处理阶段同步保存 Mel 频谱图
    print("\n[信息] 划分 Mel 频谱图数据集...")

    # 由于元数据顺序与特征提取顺序一致，用原始划分的索引
    train_size = len(data_dict["X_train"])
    val_size = len(data_dict["X_val"])

    X_train_mel = all_mel_features[:train_size]
    X_val_mel = all_mel_features[train_size:train_size + val_size]
    X_test_mel = all_mel_features[train_size + val_size:]

    # 扩展维度以适应 CNN 输入 (batch, channel, height, width)
    X_train_mel = X_train_mel[:, np.newaxis, :, :]
    X_val_mel = X_val_mel[:, np.newaxis, :, :]
    X_test_mel = X_test_mel[:, np.newaxis, :, :]

    print(f"  Mel 训练集: {X_train_mel.shape}")
    print(f"  Mel 验证集: {X_val_mel.shape}")
    print(f"  Mel 测试集: {X_test_mel.shape}")

    # 训练 CNN
    model = EmotionCNN(
        n_mels=128,
        time_steps=94,
        n_classes=len(np.unique(data_dict["y_train"])),
        dropout_rate=0.3,
    )

    trainer, result = train_and_evaluate_deep_model(
        model=model,
        X_train=X_train_mel,
        y_train=data_dict["y_train"],
        X_val=X_val_mel,
        y_val=data_dict["y_val"],
        X_test=X_test_mel,
        y_test=data_dict["y_test"],
        batch_size=training_cfg.batch_size,
        num_epochs=training_cfg.num_epochs,
        learning_rate=training_cfg.learning_rate,
        model_name="cnn_model",
    )

    return {"trainer": trainer, **result}


def main():
    """主函数：加载数据 -> 训练模型 -> 评估 -> 保存结果"""
    print("=" * 60)
    print("  语音情感识别系统 - 模型训练")
    print("=" * 60)

    # 创建保存目录
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1. 加载数据
    data_dict = load_casia_data()
    if not data_dict:
        print("[错误] 数据加载失败，请先运行预处理脚本")
        return

    # 保存所有标签用于 CNN
    all_labels = np.concatenate([
        data_dict["y_train"],
        data_dict["y_val"],
        data_dict["y_test"],
    ])
    data_dict["y_all"] = all_labels

    # 存储所有模型结果用于对比
    all_model_results = {}
    all_predictions = {}

    # 2. 训练传统 ML 模型
    ml_results = train_traditional_models(data_dict)

    for model_name, result in ml_results.items():
        y_pred = result["predictions"]
        y_test = data_dict["y_test"]

        metrics = evaluate_model_comprehensive(
            y_test, y_pred,
            model_name=f"SVM" if "svm" in model_name else "RandomForest",
            save_dir=results_dir,
            show_plots=False,
        )

        all_model_results[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
        }
        all_predictions[model_name] = y_pred

    # 3. 训练 MLP 模型
    mlp_result = train_mlp_model(data_dict)
    if mlp_result:
        y_pred = mlp_result["predictions"]
        y_test = data_dict["y_test"]

        metrics = evaluate_model_comprehensive(
            y_test, y_pred,
            model_name="MLP",
            history=mlp_result.get("history"),
            save_dir=results_dir,
            show_plots=False,
        )

        all_model_results["mlp"] = {
            "y_true": y_test,
            "y_pred": y_pred,
        }
        all_predictions["mlp"] = y_pred

    # 4. 训练 CNN 模型（可选，需要原始音频文件）
    print("\n" + "=" * 60)
    print("  是否训练 CNN 模型？（需要重新提取 Mel 频谱图）")
    print("  输入 'y' 继续，其他跳过: ", end="")
    choice = input().strip().lower()

    if choice == "y":
        cnn_result = train_cnn_model(data_dict)
        if cnn_result:
            y_pred = cnn_result["predictions"]
            y_test = data_dict["y_test"]

            metrics = evaluate_model_comprehensive(
                y_test, y_pred,
                model_name="CNN",
                history=cnn_result.get("history"),
                save_dir=results_dir,
                show_plots=False,
            )

            all_model_results["cnn"] = {
                "y_true": y_test,
                "y_pred": y_pred,
            }
            all_predictions["cnn"] = y_pred

    # 5. 模型对比
    if len(all_model_results) >= 2:
        print("\n" + "=" * 60)
        print("  模型对比")
        print("=" * 60)

        compare_and_report(
            all_model_results,
            save_dir=results_dir,
            show_plots=True,
        )

    print("\n" + "=" * 60)
    print("  训练完成！")
    print(f"  模型保存路径: {SAVED_MODELS_DIR}")
    print(f"  结果保存路径: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
