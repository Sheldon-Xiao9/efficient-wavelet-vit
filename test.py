import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from network.model import DeepfakeDetector
from config.transforms import get_transforms
from config.data_loader import FaceForensicsLoader

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepfakeDetector model")
    parser.add_argument("--root", "--r", type=str, default="/path/to/dataset", 
                        help="Dataset root directory")
    parser.add_argument("--batch-size", "--bs", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--frame-count", "--fc", type=int, default=30,
                        help="Number of frames per video")
    parser.add_argument("--dim", "--d", type=int, default=128,
                        help="Feature dimension")
    parser.add_argument("--split", "--s", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which dataset split to use")
    parser.add_argument("--sample-index", "--i", type=int, default=None,
                        help="Specific sample index to test (optional)")
    return parser.parse_args()

def test_model(args):
    """
    测试DeepfakeDetector模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*50)
    
    # 参数设置
    batch_size = 8
    frame_count = 24
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
        print("2. Loading Real Data...")
        
        transform = get_transforms()
        test_batch = FaceForensicsLoader(
            root=args.root,
            split=args.split,
            frame_count=args.frame_count,
            transform=transform['test']
        )
        print(f"Dataset size: {len(test_batch)}")
        
        # 直接从数据集获取样本，不使用DataLoader
        frames_list = []
        labels_list = []
        
        # 根据batch_size选择样本
        indices = []
        if args.sample_index is not None:
            # 如果指定了特定索引，则使用它
            indices = [args.sample_index]
        else:
            # 随机选择batch_size个样本
            indices = np.random.choice(len(test_batch), batch_size, replace=False)
        
        # 加载选定的样本
        for idx in indices:
            frames, label = test_batch[idx]
            frames_list.append(frames)
            labels_list.append(label)
        
        # 将列表转换为批次张量
        frames = torch.stack(frames_list).to(device)
        labels = torch.tensor(labels_list).to(device)
        print(f"Loaded {len(frames_list)} samples")
        print(f"Input shape: Frames - {frames.shape}; Labels - {labels.shape}")
        print("="*50)
        
        # 分步训练测试
        with torch.no_grad():
            start_time = time.time()
            
            # DAMA模块
            print("3. Testing DAMA module...")
            dama_feats = model.dama(frames, batch_size=batch_size)
            print(f"DAMA output shape: {dama_feats.shape}")
            print("="*50)
            
            # TCM模块
            print("4. Testing TCM module...")
            tcm_outputs = model.tcm(frames, dama_feats)
            print("TCM output keys: ")
            for key, value in tcm_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
            print("="*50)
            
            # 测试完整模型
            print("5. Testing complete model...")
            outputs = model(frames, batch_size=batch_size)
            print("Model output keys: ")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
            print("="*50)
            
            # 打印结果
            if 'logits' in outputs:
                probs = softmax(outputs['logits'], dim=1)
                print(f"Predicted probabilities: ")
                for i in range(len(labels)):
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
    args = parse_args()
    success = test_model(args)
    if success:
        print("="*50)
        print("Test passed successfully!")
    else:
        print("="*50)
        print("Test failed!")