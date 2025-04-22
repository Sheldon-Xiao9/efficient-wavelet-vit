import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os
import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.xception.models import model_selection
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms
from utils.visualization import TrainVisualization

# 参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Train Xception for Deepfake Detection")
    parser.add_argument("--root", "--r", type=str, default="/path/to/dataset", 
                        help="Dataset root directory")
    parser.add_argument("--output", "--o", type=str, default="./output/xception", 
                        help="Output directory")
    parser.add_argument("--batch-size", "--bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", "--e", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--frame-count", "--fc", type=int, default=1,
                        help="Number of frames per video")
    parser.add_argument("--visualize", "--v", action="store_true",
                        help="Generate visualizations after training is done")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability")
    parser.add_argument("--multi-gpu", "--mg", action="store_true",
                        help="Use multiple GPUs for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个轮次
    """
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    
    for i, (frames, labels) in enumerate(tqdm(dataloader, desc="Training iteration")):
        batch_loss = 0.0
        batch_size = frames.size(0)
        labels = labels.to(device).float()
        
        if frames.dim() == 5:  # [batch, frames, channels, height, width]
            # 处理所有帧并聚合结果
            frame_count = frames.size(1)
            all_outputs = []
            
            # 处理批次中的每个样本的所有帧
            for b in range(batch_size):
                sample_outputs = []
                for f in range(frame_count):
                    frame = frames[b, f].unsqueeze(0).to(device)  # [1, C, H, W]
                    output = model(frame)
                    
                    # 处理元组输出
                    if isinstance(output, tuple):
                        output = output[1]  # 分类输出
                    
                    sample_outputs.append(output)
                
                # 对单个样本的所有帧结果进行平均
                sample_pred = torch.mean(torch.cat(sample_outputs, dim=0), dim=0, keepdim=True)
                all_outputs.append(sample_pred)
            
            # 合并批次中所有样本的预测
            outputs = torch.cat(all_outputs, dim=0)
            
        else:  # 单帧输入 [batch, channels, height, width]
            frames = frames.to(device)
            outputs = model(frames)
            
            # 处理元组输出
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # 分类输出
            
        # 计算损失和反向传播
        loss = criterion(outputs, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        
        # 分类预测
        preds = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_acc = accuracy_score(labels_all, [1 if p >= 0.5 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc
    }
    
def val_epoch(model, dataloader, criterion, device):
    """
    验证一个轮次
    """
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for frames, labels in dataloader:
            batch_size = frames.size(0)
            labels = labels.to(device).float()
            
            if frames.dim() == 5:  # [batch, frames, channels, height, width]
                # 处理所有帧并聚合结果
                frame_count = frames.size(1)
                all_outputs = []
                
                # 处理批次中的每个样本的所有帧
                for b in range(batch_size):
                    sample_outputs = []
                    for f in range(frame_count):
                        frame = frames[b, f].unsqueeze(0).to(device)  # [1, C, H, W]
                        output = model(frame)
                        
                        # 处理元组输出
                        if isinstance(output, tuple):
                            output = output[1]  # 分类输出
                        
                        sample_outputs.append(output)
                    
                    # 对单个样本的所有帧结果进行平均
                    sample_pred = torch.mean(torch.cat(sample_outputs, dim=0), dim=0, keepdim=True)
                    all_outputs.append(sample_pred)
                
                # 合并批次中所有样本的预测
                outputs = torch.cat(all_outputs, dim=0)
                
            else:  # 单帧输入 [batch, channels, height, width]
                frames = frames.to(device)
                outputs = model(frames)
                
                # 处理元组输出
                if isinstance(outputs, tuple):
                    outputs = outputs[1]  # 分类输出
                
            loss = criterion(outputs, labels.unsqueeze(1))
            running_loss += loss.item() * batch_size
            
            # 分类预测
            preds = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_acc = accuracy_score(labels_all, [1 if p >= 0.5 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc
    }
    
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("Start setting...")
    
    # 检查可用的GPU
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")

    print("="*50)
    
    # 数据加载器
    print("Initializing data loaders...")
    transforms = get_transforms()
    
    train_dataset = FaceForensicsLoader(
        root=args.root,
        split='train',
        frame_count=args.frame_count,
        transform=transforms['train']
    )
    
    val_dataset = FaceForensicsLoader(
        root=args.root,
        split='val',
        frame_count=args.frame_count,
        transform=transforms['val']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
   
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print("="*50)
    print(f"Model Initialization...")
    
    # 初始化模型
    model, img_size, is_pretrained, input_list, _ = model_selection(
        'xception', num_out_classes=1, dropout=args.dropout
    )
    model = model.to(device)
    
    if args.multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
    
    print("Hyperparameters:")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Frame count: {args.frame_count}")
    print(f"Input size: ({img_size}, {img_size})")
    print(f"Dropout: {args.dropout}")
    print("Model initialized successfully!")
    
    print("="*50)
    print("Start training...")
    print("="*50)
    
    train_viz = TrainVisualization(os.path.join(args.output, 'train_visualizations'))
    
    # 计算正样本权重
    real_count = len(train_dataset.real_videos)
    fake_count = len(train_dataset.fake_videos)
    alpha = torch.tensor([fake_count / (real_count + fake_count)]).to(device)
    
    criterion = BCEWithLogitsLoss(pos_weight=alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    best_val_auc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")
        
        # 更新采样策略
        train_dataset.update_sampling_strategy(epoch, args.epochs)
        val_dataset.update_sampling_strategy(epoch, args.epochs)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_time = time.time()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # 验证
        val_metrics = val_epoch(model, val_loader, criterion, device)
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print(f"New best model saved with AUC: {best_val_auc}")
            
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
        }, os.path.join(args.output, f'checkpoint_{epoch+1}.pth'))
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['auc']:.4f} | "
              f"Train ACC: {train_metrics['acc']:.4f} | "
              f"Time: {epoch_time:.2f}s")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val ACC: {val_metrics['acc']:.4f}")
        
        train_viz.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[0]['lr']
        )
        
        # 保存训练可视化
        train_viz.save_metrics()
        
        print("="*50)
    
    # 如果需要，生成可视化
    if args.visualize:
        train_viz.plot_all()
        
if __name__ == "__main__":
    main()