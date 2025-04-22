import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（往上两级）
project_root = os.path.dirname(os.path.dirname(current_dir))

# 将项目根目录添加到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 直接执行train.py而不是导入它
if __name__ == "__main__":
    # 获取train.py的完整路径
    train_path = os.path.join(current_dir, "train.py")
    
    # 执行train.py
    with open(train_path, 'r') as file:
        exec(compile(file.read(), train_path, 'exec'))