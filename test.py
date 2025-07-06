import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import torch
import time
import numpy as np
from network.model import DeepfakeDetector

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Test DeepfakeDetector model for inference time")
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to the model weights file (.pth)")
    parser.add_argument("--batch-size", "--bs", type=int, default=1,
                        help="Batch size for inference. Default is 1.")
    parser.add_argument("--frame-count", "--fc", type=int, default=30,
                        help="Number of frames per video")
    parser.add_argument("--dim", "--d", type=int, default=128,
                        help="Feature dimension")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of inference runs to average for timing")
    return parser.parse_args()

def test_inference_time(args):
    """
    测试DeepfakeDetector模型的推理时间
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("="*50)
    
    # 参数设置
    batch_size = args.batch_size if args.batch_size else 1 if args.batch_size else 1
    frame_count = args.frame_count if args.frame_count else 1 if args.frame_count else 1
    in_channels = 3
    dim = args.dim if args.dim else 128 if args.dim else 128
    input_size = (224, 224)
    
    print(f"Batch size: {batch_size}")
    print(f"Frame count: {frame_count}")
    print(f"Feature dimension: {dim}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Number of runs for timing: {args.runs}")
    print("="*50)
    
    # 初始化模型
    try:
        print("1. Initializing model...")
        model = DeepfakeDetector(in_channels=in_channels, dama_dim=dim, batch_size=batch_size)
        
        print(f"2. Loading weights from '{args.weights}'...")
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print("Weights loaded successfully!")
        
        model.to(device)
        model.eval()  # 设置为评估模式
        
        # 统计模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("="*50)
        
        # 创建随机输入
        print("3. Creating random input for inference...")
        frames = torch.randn(batch_size, frame_count, in_channels, input_size[0], input_size[1]).to(device)
        print(f"Input tensor shape: {frames.shape}")
        print("="*50)
        
        # 预热
        print("4. Warming up the model for accurate timing...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(frames, batch_size=batch_size, ablation='dynamic')
        
        # 测试完整模型推理时间
        print(f"5. Running inference {args.runs} times to measure performance...")
        total_time = 0
        with torch.no_grad():
            for i in range(args.runs):
                torch.cuda.synchronize() # 等待CUDA操作完成
                start_time = time.time()
                
                _ = model(frames, batch_size=batch_size, ablation='dynamic')
                
                torch.cuda.synchronize() # 再次同步以确保计时准确
                end_time = time.time()
                
                total_time += (end_time - start_time)
        
        average_time = total_time / args.runs
        fps = 1 / average_time
        
        print("="*50)
        print("Inference Time Results:")
        print(f"  - Total time for {args.runs} runs: {total_time:.4f} seconds")
        print(f"  - Average inference time per run: {average_time * 1000:.4f} ms")
        print(f"  - Frames Per Second (FPS): {fps:.2f}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    args = parse_args()
    success = test_inference_time(args)
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
