import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
# from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score # type: ignore

from network.model import DeepfakeDetector
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms
from config.focal_loss import BinaryFocalLoss
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def orthogonal_loss(space_feats, freq_feats):
    """
    正交约束损失
    """
    _, feat_dim = space_feats.shape
    space_feats = F.normalize(space_feats, p=2, dim=1)
    freq_feats = F.normalize(freq_feats, p=2, dim=1)
    
    # 计算协方差矩阵并惩罚非对角线元素
    cov = torch.mm(space_feats.T, freq_feats)  # [feat_dim, feat_dim]
    diag_mask = torch.eye(feat_dim).to(cov.device)
    off_diag = cov * (1 - diag_mask)
    return torch.norm(off_diag, p='fro')**2 / (feat_dim*(feat_dim-1))
    
def combined_loss(outputs, labels, criterion, epoch, max_epochs):
    """
    组合损失函数，由Focal Loss和正交约束组成
    """
    logits = outputs['logits']
    labels = labels.view(-1, 1).float()
    
    if epoch < 0.2 * max_epochs:
        cls_loss = criterion(logits, labels)
        return cls_loss, {
            'cls_loss': cls_loss.item(),
            'orth_loss': 0.0
        }
    else:
        cls_loss = criterion(logits, labels)
        # 正交约束
        loss_orth = orthogonal_loss(outputs['space'], outputs['freq'])
        lambda_orth  = min(1.0, (epoch - 0.2 * max_epochs) / (0.5 * max_epochs))
    
    return cls_loss + lambda_orth * loss_orth, {
        'cls_loss': cls_loss.item(),
        'orth_loss': loss_orth.item()
    }

def train_epoch(model, dataloader, criterion, optimizer, device, batch_size, accum_steps=2, epoch=None, max_epochs=None):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    preds_all, labels_all = [], []
    
    optimizer.zero_grad()
    
    for i, (frames, labels) in enumerate(tqdm(dataloader, desc="Training iteration")):
        frames, labels = frames.to(device), labels.to(device)
        
        outputs = model(frames, batch_size=batch_size, ablation='dynamic')
        
        loss, losses = combined_loss(outputs, labels, criterion, epoch, max_epochs)
        
        # 梯度累积
        orig_loss = loss
        loss = loss / accum_steps
        loss.backward()
        
        if (i+1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += orig_loss.item() * frames.size(0)
        running_cls_loss += losses['cls_loss'] * frames.size(0)
        
        # 分类预测
        preds = torch.sigmoid(outputs['logits']).squeeze(1).detach().cpu().numpy()
        preds_all.extend(preds)
        labels_all.extend(labels.cpu().numpy())
        
    if len(dataloader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_f1 = f1_score(labels_all, [1 if p >= 0.5 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'auc': epoch_auc,
        'f1': epoch_f1
    }
    
def val_epoch(model, dataloader, criterion, device, batch_size, epoch=None, max_epochs=None):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames, batch_size=batch_size, ablation='dynamic')
            loss, losses = combined_loss(outputs, labels, criterion, epoch, max_epochs)
            
            running_loss += loss.item() * frames.size(0)
            running_cls_loss += losses['cls_loss'] * frames.size(0)
            
            # 分类预测
            preds = torch.sigmoid(outputs['logits']).squeeze(1).detach().cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_f1 = f1_score(labels_all, [1 if p >= 0.5 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'auc': epoch_auc,
        'f1': epoch_f1
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
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 备份验证集视频
    val_fake_backup = copy.deepcopy(val_dataset.fake_videos)
    val_real_backup = copy.deepcopy(val_dataset.real_videos)
    
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
    print("Input size: (224, 224)")
    print("Model initialized successfully!")
    
    print("="*50)
    print("Start training...")
    print("="*50)
    
    train_viz = TrainVisualization(os.path.join(args.output, 'train_visualizations'))
    
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    best_val_auc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")
        
        print(f"Resampling fake videos...")
        train_dataset.resample_fake_videos(epoch, args.epochs)
        
        # 每轮开始前恢复验证集到原始状态
        val_dataset.fake_videos = copy.deepcopy(val_fake_backup)
        val_dataset.real_videos = copy.deepcopy(val_real_backup)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        start_time = time.time()
        
        # 如果训练已超过 60% 的 Epochs，则解冻 EfficientNet 和 ViT 参数
        if epoch >= 0.6 * args.epochs:
            base_model = model.module if args.multi_gpu else model
            for param in base_model.sfe.space_efficient.parameters():
                param.requires_grad = True
            print("Unfreezing EfficientNet parameters...")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, args.batch_size, args.accum_steps, epoch, args.epochs)
        scheduler.step()
        
        # 验证
        with torch.no_grad():
            val_metrics = val_epoch(model, val_loader, criterion, device, args.batch_size, epoch, args.epochs)
        
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
        }, os.path.join(args.output, f'checkpoint_{epoch+1}.pth'))
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train AUC: {train_metrics['auc']:.4f} | "
              f"Train F1 Score: {train_metrics['f1']:.4f} | "
              f"Time: {epoch_time:.2f}s")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1 Score: {val_metrics['f1']:.4f}")
        
        train_viz.update(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[0]['lr']
        )
        
        # 保存训练可视化
        train_viz.save_metrics()
        
        print("="*50)
    
    # 如果有可视化参数，则生成可视化结果
    if args.visualize:
        train_viz.plot_all()
        
if __name__ == "__main__":
    main()