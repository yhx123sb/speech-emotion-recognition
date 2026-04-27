"""快速测试 GPU 训练"""
import torch
import numpy as np
import time
from speech_emotion_recognition.models.cnn_model import EmotionMLP
from speech_emotion_recognition.models.trainer import Trainer

np.random.seed(42)
X_train = np.random.randn(840, 122).astype(np.float32)
y_train = np.random.randint(0, 6, 840)
X_val = np.random.randn(120, 122).astype(np.float32)
y_val = np.random.randint(0, 6, 120)

model = EmotionMLP(input_dim=122, n_classes=6)
trainer = Trainer(model=model, learning_rate=0.001)
print(f"Device: {trainer.device}")

start = time.time()
history = trainer.train(X_train, y_train, X_val, y_val, batch_size=32, num_epochs=10, verbose=True)
elapsed = time.time() - start
print(f"Total time: {elapsed:.2f}s")
print(f"Best val acc: {max(history['val_acc']):.4f}")
