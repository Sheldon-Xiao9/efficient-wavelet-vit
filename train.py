import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score # type: ignore

from network.model import DeepfakeDetector
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms
from utils.visualization import TrainVisualization

# 参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--root", "--r", type=str, default="/path/to/dataset", 
                        help="Dataset root directory")
    parser.add_argument("--output", "--o", type=str, default="./output", 
                        help="Output directory")
    parser.add_argument("--batch-size", "--bs", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", "--e", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dim", "--d", type=int, default=128,
                        help="Feature dimension")
    parser.add_argument("--frame-count", "--fc", type=int, default=30,
                        help="Number of frames per video")
    parser.add_argument("--visualize", "--v", action="store_true",
                        help="Generate visualizations after training is done")
    parser.add_argument("--accum-steps", "--as", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--multi-gpu", "--mg", action="store_true",
                        help="Use multiple GPUs for training")
    parser.add_argument("--parallel", "--p", action="store_true",
                        help="Start frame-level parallelism")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def combined_loss(outputs, labels, criterion, alpha=0.7, beta=0.3):
    """
    组合损失函数，由交叉熵损失和时序一致性得分构成
    """
    classification_loss = criterion(outputs['logits'], labels)
    
    consistency = outputs['tcm_consistency']
    
    consistency = torch.sigmoid(consistency)
    
    # 根据标签设置目标一致性值
    # 真实视频(lable=0)的一致性得分应该接近1
    # 伪造视频(label=1)的一致性得分应该接近0
    target_consistency = 1.0 - labels.float()
    consistency_loss = F.mse_loss(consistency, target_consistency)
    
    # 加权组合
    combined = alpha * classification_loss + beta * consistency_loss
    return combined, {
        'cls_loss': classification_loss.item(),
        'cons_loss': consistency_loss.item(),
    }

def train_epoch(model, dataloader, criterion, optimizer, device, batch_size, accum_steps=2):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0
    preds_all, labels_all = [], []
    
    optimizer.zero_grad()
    
    for i, (frames, labels) in enumerate(tqdm(dataloader, desc="Training iteration")):
        frames, labels = frames.to(device), labels.to(device)
        
        outputs = model(frames, batch_size=batch_size)
        
        loss, losses = combined_loss(outputs, labels, criterion)
        
        # 梯度累积
        loss = loss / accum_steps
        loss.backward()
        
        if (i+1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Batch {i+1}/{len(dataloader)}: Loss: {loss.item():.4f}")
        
        running_loss += loss.item() * frames.size(0)
        running_cls_loss += losses['cls_loss'] * frames.size(0)
        running_cons_loss += losses['cons_loss'] * frames.size(0)
        
        # 分类预测
        preds = torch.softmax(outputs['logits'], dim=1)[:, 1].detach().cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())
        
    if len(dataloader.dataset) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
    epoch_cons_loss = running_cons_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_acc = accuracy_score(labels_all, [1 if p >= 0.55 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cons_loss': epoch_cons_loss,
        'auc': epoch_auc,
        'acc': epoch_acc
    }
    
def val_epoch(model, dataloader, criterion, device, batch_size):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_cons_loss = 0.0
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames, batch_size=batch_size)
            loss, losses = combined_loss(outputs, labels, criterion)
            
            running_loss += loss.item() * frames.size(0)
            running_cls_loss += losses['cls_loss'] * frames.size(0)
            running_cons_loss += losses['cons_loss'] * frames.size(0)
            
            # 分类预测
            preds = torch.softmax(outputs['logits'], dim=1)[:, 1].detach().cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
    epoch_cons_loss = running_cons_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_acc = accuracy_score(labels_all, [1 if p >= 0.55 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'cons_loss': epoch_cons_loss,
        'auc': epoch_auc,
        'acc': epoch_acc
    }
    
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("Start setting...")
    
    # 检查可用GPU数量
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
    # 获取数据转换
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
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print("="*50)
    print(f"Model Initialization...")
    
    model = DeepfakeDetector(
        in_channels=3,
        dama_dim=args.dim,
        batch_size=args.batch_size
    ).to(device)
    
    if args.multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
    
    print("Hyperparameters:")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Feature dimension: {args.dim}")
    print(f"Frame count: {args.frame_count}")
    print("Input size: (384, 384)")
    print("Model initialized successfully!")
    
    print("="*50)
    print("Start training...")
    print("="*50)
    
    train_viz = TrainVisualization(os.path.join(args.output, 'train_visualizations'))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_auc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.batch_size, args.accum_steps)
        scheduler.step()
        
        # 验证
        with torch.no_grad():
            val_metrics = val_epoch(model, val_loader, criterion, device, args.batch_size)
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print(f"New best model saved with AUC: {best_val_auc}")
            
        # 检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
        }, os.path.join(args.output, 'checkpoint.pth'))
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['auc']:.4f} | "
              f"Train Accuracy: {train_metrics['acc']:.4f} | "
              f"Time: {epoch_time:.2f}s")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val Accuracy: {val_metrics['acc']:.4f}")
        
        train_viz.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[0]['lr']
        )
        print("="*50)
    
    # 如果有可视化参数，则生成可视化结果
    if args.visualize:
        train_viz.plot_all()
        
if __name__ == "__main__":
    main()