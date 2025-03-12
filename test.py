import torch
import time
import numpy as np
from torch.nn.functional import softmax
import os
from network.model import DeepfakeDetector

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def test_model():
    """
    测试DeepfakeDetector模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*50)
    
    # 参数设置
    batch_size = 8
    frame_count = 30
    in_channels = 3
    dim = 128
    input_size = (224, 224)
    
    print(f"Batch size: {batch_size}")
    print(f"Frame count: {frame_count}")
    print(f"Input channels: {in_channels}")
    print(f"Feature dimension: {dim}")
    print(f"Input size: {input_size} x {input_size}")
    print("="*50)
    
    # 初始化模型
    try:
        print("1. Initializing model...")
        model = DeepfakeDetector(in_channels=in_channels, dama_dim=dim, batch_size=batch_size)
        model.to(device)
        print("Model initialized successfully!")
        
        # 统计模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print("="*50)
        
        # 创建随机输入
        print("2. Creating random input...")
        x = torch.randn(batch_size, frame_count, in_channels, *input_size).to(device)
        print(f"Input shape: {x.shape}")
        print("="*50)
        
        # 分步训练测试
        with torch.no_grad():
            start_time = time.time()
            
            # DAMA模块
            print("3. Testing DAMA module...")
            dama_feats = model.dama(x, batch_size=batch_size)
            print(f"DAMA output shape: {dama_feats.shape}")
            print("="*50)
            
            # TCM模块
            print("4. Testing TCM module...")
            tcm_outputs = model.tcm(x, dama_feats)
            print("TCM output keys: ")
            for key, value in tcm_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
            print("="*50)
            
            # 测试完整模型
            print("5. Testing complete model...")
            outputs = model(x, batch_size=batch_size)
            print("Model output keys: ")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
            print("="*50)
            
            # 打印结果
            if 'logits' in outputs:
                probs = softmax(outputs['logits'], dim=1)
                print(f"Predicted probabilities: ")
                for i in range(batch_size):
                    print(f"  - Sample {i+1}: Real: {probs[i, 0].item():.4f}, Fake: {probs[i, 1].item():.4f}")
                print("="*50)
            
            end_time = time.time()
            print(f"Time elapsed: {end_time - start_time:.2f} seconds")
            
            return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    success = test_model()
    if success:
        print("="*50)
        print("Test passed successfully!")
    else:
        print("="*50)
        print("Test failed!")