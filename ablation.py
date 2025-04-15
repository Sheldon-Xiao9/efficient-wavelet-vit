import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve # type: ignore
from tqdm import tqdm
import yaml # type: ignore
import pandas as pd # type: ignore
from datetime import datetime

# 导入模型与数据集定义文件
from network.model import DeepfakeDetector
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Deepfake Detection Ablation Study')
    parser.add_argument('--root', type=str, default='/path/to/dataset', help='Path to the dataset')
    parser.add_argument('--output', type=str, default='./output/ablation', help='Path to save the ablation output')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the DAMA module')
    parser.add_argument('--frame-count', type=int, default=24, help='Number of frames per video')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, ablation_mode, epochs, video_batch_size):
    """训练模型并返回训练过程中的指标"""
    train_losses = []
    train_accs = []
    train_aucs = []
    val_losses = []
    val_accs = []
    val_aucs = []
    
    best_val_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 训练模式
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for frames, labels in tqdm(train_loader, desc=f"Training {ablation_mode}"):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames, batch_size=video_batch_size, ablation=ablation_mode)
            logits = outputs['logits']
            
            loss = criterion(logits.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * frames.size(0)
            
            # 计算预测
            preds = torch.sigmoid(logits).cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # 计算训练指标
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        epoch_auc = roc_auc_score(all_labels, all_preds)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        train_aucs.append(epoch_auc)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, criterion, device, ablation_mode, video_batch_size)
        
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        val_aucs.append(val_metrics['auc'])
        
        # 更新学习率
        scheduler.step()
        
        # 打印当前指标
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
        
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    metrics = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'train_aucs': train_aucs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }
    
    return model, metrics

def evaluate_model(model, dataloader, criterion, device, ablation_mode, video_batch_size):
    """评估模型性能"""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc=f"评估 {ablation_mode}"):
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames, batch_size=video_batch_size, ablation=ablation_mode)
            logits = outputs['logits']
            
            loss = criterion(logits.squeeze(), labels.float())
            val_loss += loss.item() * frames.size(0)
            
            scores = torch.sigmoid(logits).squeeze().cpu().numpy()
            preds = (scores >= 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
    
    # 计算指标
    loss = val_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_scores)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'labels': all_labels,
        'scores': all_scores
    }
    
    return metrics

def plot_learning_curves(results, output_dir):
    """绘制学习曲线"""
    plt.figure(figsize=(18, 12))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    for ablation_mode, metrics in results.items():
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        plt.plot(metrics['train_losses'], label=f"{display_name} - Train")
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
    plt.subplot(2, 2, 2)
    for ablation_mode, metrics in results.items():
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        plt.plot(metrics['val_losses'], label=f"{display_name} - Validation")
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # AUC曲线
    plt.subplot(2, 2, 3)
    for ablation_mode, metrics in results.items():
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        plt.plot(metrics['train_aucs'], label=f"{display_name} - Train")
    plt.title('Training AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for ablation_mode, metrics in results.items():
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        plt.plot(metrics['val_aucs'], label=f"{display_name} - Validation")
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

def plot_roc_curves(test_results, output_dir):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    for ablation_mode, metrics in test_results.items():
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['scores'])
        auc = metrics['auc']
        plt.plot(fpr, tpr, label=f'{display_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
def create_metrics_table(test_results, output_dir):
    """创建指标比较表格"""
    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'F1 Score', 'AUC'])
    
    for i, (ablation_mode, metrics) in enumerate(test_results.items()):
        display_name = 'DAMA (Full)' if ablation_mode == 'dynamic' else ablation_mode
        metrics_df.loc[i] = [
            display_name,
            metrics['accuracy'],
            metrics['precision'],
            metrics['f1'],
            metrics['auc']
        ]
    
    # 保存到CSV
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
    
    return metrics_df

def ablation_experiment():
    """运行消融实验"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    transforms = get_transforms()
    
    # 加载数据集
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
    
    test_dataset = FaceForensicsLoader(
        root=args.root,
        split='test',
        frame_count=args.frame_count,
        transform=transforms['val']
    )
    
    # 创建数据加载器
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 定义消融配置
    ablation_modes = ['sfe_only', 'sfe_mwt', 'dynamic']
    
    # 存储训练和测试结果
    training_results = {}
    test_results = {}
    
    # 为每个消融配置训练和评估模型
    for ablation_mode in ablation_modes:
        print(f"\n{'='*50}")
        print(f"Starting training for {ablation_mode} mode")
        print(f"{'='*50}")
        
        # 创建模型
        model = DeepfakeDetector(
            in_channels=3,
            dama_dim=args.dim,
            batch_size=args.batch_size
        ).to(device)
        
        # 定义损失函数
        criterion = nn.BCEWithLogitsLoss()
        
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        
        # 训练模型
        model, metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            ablation_mode=ablation_mode,
            epochs=args.epochs,
            video_batch_size=args.batch_size
        )
        
        # 存储训练结果
        training_results[ablation_mode] = metrics
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(output_dir, f'{ablation_mode}_model.pth'))
        
        # 在测试集上评估模型
        print(f"\nEvaluating {ablation_mode} model on test set")
        test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            ablation_mode=ablation_mode,
            video_batch_size=args.batch_size
        )
        
        # 存储测试结果
        test_results[ablation_mode] = test_metrics
        
        # 打印测试指标
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    # 绘制学习曲线
    plot_learning_curves(training_results, output_dir)
    
    # 绘制ROC曲线
    plot_roc_curves(test_results, output_dir)
    
    # 创建指标比较表格
    metrics_df = create_metrics_table(test_results, output_dir)
    
    # 打印比较结果
    print("\n消融实验结果比较:")
    print(metrics_df)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    print(f"\nablation study completed. Results saved to {output_dir}")
    
if __name__ == "__main__":
    ablation_experiment()