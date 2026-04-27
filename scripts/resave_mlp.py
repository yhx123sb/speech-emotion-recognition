"""重新保存 MLP 模型 - 直接使用正确的 num_classes"""
import sys
sys.path.insert(0, 'D:/yuyinshibei')

# Force clear all cached modules
for key in list(sys.modules.keys()):
    if 'speech' in key or 'cnn_model' in key or 'trainer' in key:
        del sys.modules[key]

from pathlib import Path
import torch
import numpy as np

# Directly import with clean slate
from speech_emotion_recognition.models.cnn_model import EmotionMLP
from speech_emotion_recognition.models.trainer import Trainer
from speech_emotion_recognition.utils.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR

# 加载数据
X_train = np.load(PROCESSED_DATA_DIR / 'casia_X_train.npy')
y_train = np.load(PROCESSED_DATA_DIR / 'casia_y_train.npy')
X_val = np.load(PROCESSED_DATA_DIR / 'casia_X_val.npy')
y_val = np.load(PROCESSED_DATA_DIR / 'casia_y_val.npy')

# 连续化标签
all_y = np.concatenate([y_train, y_val])
unique_labels = np.sort(np.unique(all_y))
label_map = {old: new for new, old in enumerate(unique_labels)}
y_train = np.array([label_map[y] for y in y_train])
y_val = np.array([label_map[y] for y in y_val])

n_classes = len(unique_labels)

# 创建模型
model = EmotionMLP(input_dim=122, n_classes=n_classes, hidden_dims=[512, 256, 128], dropout_rate=0.3)
print(f"model.n_classes = {model.n_classes}")
trainer = Trainer(model=model, learning_rate=0.001)

# 快速训练
history = trainer.train(X_train, y_train, X_val, y_val, batch_size=32, num_epochs=10)

# 保存模型
trainer.save_model(SAVED_MODELS_DIR / 'mlp_model.pt')

# 验证
checkpoint = torch.load(str(SAVED_MODELS_DIR / 'mlp_model.pt'), map_location='cpu')
print(f"Saved num_classes: {checkpoint.get('num_classes', 'NOT FOUND')}")
print(f"Output layer shape: {checkpoint['model_state_dict']['network.12.weight'].shape}")
print("Done!")
