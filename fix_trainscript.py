"""修复训练脚本，添加标签连续化"""
import sys
sys.path.insert(0, '.')

filepath = 'scripts/train_models.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 在 "return data_dict" 之前插入标签连续化代码
old_text = """    print(f"[完成] 数据加载成功!")
    print(f"  训练集: {len(data_dict['X_train'])} 样本")
    print(f"  验证集: {len(data_dict['X_val'])} 样本")
    print(f"  测试集: {len(data_dict['X_test'])} 样本")

    return data_dict"""

new_text = """    print(f"[完成] 数据加载成功!")
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

    return data_dict"""

if old_text in content:
    content = content.replace(old_text, new_text)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("修复成功！标签连续化代码已添加。")
else:
    print("未找到匹配的文本。可能代码已被修改过。")
    # 检查是否已存在 label_map
    if 'label_map' in content:
        print("label_map 已存在于文件中。")
