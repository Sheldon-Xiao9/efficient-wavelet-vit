import os
import sys

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 导入并运行F3Net训练模块
from utils.f3net import train

# 执行训练主函数
if __name__ == "__main__":
    if hasattr(train, 'main'):
        train.main()
    else:
        # 如果没有main函数，执行train.py中的__main__部分
        import runpy
        runpy.run_module('utils.f3net.train', run_name='__main__')