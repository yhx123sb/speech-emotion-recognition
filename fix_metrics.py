"""修复 metrics.py 中的 print_confusion_matrix_text 方法"""
import sys
sys.path.insert(0, '.')

filepath = 'speech_emotion_recognition/evaluation/metrics.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到函数起始和结束
func_start = None
func_end = None
next_def_line = None

for i, line in enumerate(lines):
    if 'def print_confusion_matrix_text' in line:
        func_start = i
    if func_start is not None and func_end is None and i > func_start:
        if line.strip().startswith('def ') and 'print_confusion_matrix_text' not in line:
            func_end = i
            break

print(f"Function starts at line {func_start}, ends at line {func_end}")

# 构建新函数体
new_func = '''    def print_confusion_matrix_text(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        以文本形式打印混淆矩阵，只显示实际出现的类别
        """
        cm = self.confusion_matrix(y_true, y_pred)
        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        present_names = [self.emotion_labels[i] for i in present_labels if i < len(self.emotion_labels)]

        print(f"\\n  混淆矩阵:")
        print(f"  {chr(39)}{chr(39)}:>12", end="")
        for label in present_names:
            print(f"{label:>10}", end="")
        print()

        for i, label in enumerate(present_names):
            print(f"  {label:>10}:", end="")
            for j in range(len(present_names)):
                print(f"{cm[i, j]:>10}", end="")
            print()

'''

if func_start is not None and func_end is not None:
    new_lines = lines[:func_start] + [new_func] + lines[func_end:]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"Replaced lines {func_start} to {func_end}")
else:
    print("Could not find function boundaries")
