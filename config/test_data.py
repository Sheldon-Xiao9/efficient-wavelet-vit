import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid # type: ignore
import cv2
from tqdm import tqdm
import argparse

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms

def show_batch(batch, title=None, save_path=None):
    """
    显示一个batch的图像
    """
    # 将图像转换为numpy数组
    grid = make_grid(batch, nrow=8, padding=2, pad_value=1)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    # 显示图像
    plt.figure(figsize=(20, 20))
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Saved batch visualization to '{save_path}'")
    plt.show()
    
def test_data_loader(args):
    """
    测试数据加载器
    """
    print("1. Initializing data loader...")
    
    # 设置转换
    transform = get_transforms(args.method)
    print(f"Transform: {args.method}")
    
    # 创建数据集
    dataset = FaceForensicsLoader(
        root=args.root,
        split=args.split,
        frame_count=args.frame_count,
        transform=transform,
        compression=args.compression,
        methods=[args.method]
    )
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Data loader initialized successfully!")
    
    real_count = 0
    fake_count = 0
    frame_shapes = []
    empty_frames = 0
    
    for i, (frames, labels) in enumerate(tqdm(loader, desc="加载批次")):
        if i >= args.num_batches:
            break
            
        # 统计标签分布
        batch_real = (labels == 0).sum().item()
        batch_fake = (labels == 1).sum().item()
        real_count += batch_real
        fake_count += batch_fake
        
        # 检查帧的形状
        frame_shapes.append(frames.shape)
        
        # 检查空帧
        if frames.numel() == 0:
            empty_frames += 1
            print("警告: 检测到空帧!")
            continue
        
        # 可视化
        if args.visualize:
            # 展示批次中的第一个视频的第一帧
            first_frames = frames[0, 0]
            save_path = os.path.join(args.output_dir, f"batch_{i}_frames.png")
            
            # 如果是张量，转换为PIL图像进行显示
            if isinstance(first_frames, torch.Tensor):
                show_batch(frames[:, 0], 
                          title=f"Batch {i}, Labels: {labels.tolist()}", 
                          save_path=save_path)
                print(f"Initial tensor shape: {first_frames.shape}")
            
            # 逐个检查帧中是否有人脸
            if args.check_faces:
                for b in range(min(2, frames.shape[0])):
                    video_frames = frames[b]
                    video_label = labels[b].item()
                    
                    # 保存原始帧用于人脸检测
                    frame_np = (video_frames[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    
                    # 使用OpenCV检测人脸
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    # 在帧上绘制人脸框
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # 保存标记了人脸的图像
                    face_path = os.path.join(args.output_dir, f"batch_{i}_video_{b}_face.png")
                    cv2.imwrite(face_path, frame_bgr)
                    print(f"Video {b}, Label: {video_label}, Faces detected: {len(faces)}")
                    
    # 打印统计信息
    print("\nDataset statistics:")
    print(f"Real samples: {real_count}, Fake samples: {fake_count}")
    print(f"Frame shapes(min): {min(frame_shapes, key=lambda x: x[2])}")
    print(f"Frame shapes(max): {max(frame_shapes, key=lambda x: x[2])}")
    print(f"Empty frames: {empty_frames}")
    
    if args.verbose:
        print(f"\nVideos detailed information:")
        real_paths = dataset.real_videos[:3]
        fake_paths = dataset.fake_videos[:3]
        
        print("Real videos(first 3 paths):")
        for path in real_paths:
            print(path)
        
        print("Fake videos(first 3 paths):")
        for path in fake_paths:
            print(path)
        
def main():
    # 设置参数
    parser = argparse.ArgumentParser(description="FaceForensics data loader test")
    parser.add_argument("--root", type=str, default="/kaggle/input", help="Data root directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Data split")
    parser.add_argument("--frame_count", type=int, default=30, help="Number of frames to load")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--compression", type=int, default="c23", choices=["raw", "c23", "c40"], help="Compression method")
    parser.add_argument("--method", type=str, default="DeepFake", choices=["DeepFake", "Face2Face", "FaceSwap", "NeuralTextures"], help="Method to load")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    parser.add_argument("--visualize", action="store_true", help="Visualize data")
    parser.add_argument("--check_faces", action="store_true", help="Check if there are faces in frames")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to load")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行测试
    test_data_loader(args)
    
if __name__ == "__main__":
    main()