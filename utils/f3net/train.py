import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.f3net.utils import evaluate, setup_logger
from utils.f3net.trainer import Trainer
from config.data_loader import FaceForensicsLoader
from config.transforms import get_transforms

# 原有的超参数配置
dataset_path = '/root/'
pretrained_path = 'pretrained/xception-b5690688.pth'
batch_size = 8
gpu_ids = [*range(osenvs)]
max_epoch = 30
loss_freq = 40
mode = 'FAD' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = 'output/f3net'
ckpt_name = 'FAD4_bz128'
frame_num = 24

def train_epoch(model, dataloader, device, epoch):
    """
    训练一个轮次
    """
    model.model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    
    for i, (data, label) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        data, label = data.to(device), label.to(device).float()
        
        # 设置输入并优化
        model.set_input(data, label)
        loss = model.optimize_weight()
        
        # 更新步数计数器
        model.total_steps += 1
        
        running_loss += loss.item() * data.size(0)
        
        # 收集预测和标签
        with torch.no_grad():
            _, output = model.model(data)
            preds = torch.sigmoid(output).squeeze(1).detach().cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(label.cpu().numpy())
        
        # 打印损失
        if model.total_steps % loss_freq == 0:
            print(f'loss: {loss.item()} at step: {model.total_steps}')
    
    # 计算指标
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_auc = roc_auc_score(labels_all, preds_all)
    epoch_acc = accuracy_score(labels_all, [1 if p >= 0.5 else 0 for p in preds_all])
    
    return {
        'loss': epoch_loss,
        'auc': epoch_auc,
        'acc': epoch_acc
    }

def val_epoch(model, dataset_path, device, mode='valid'):
    """
    评估模型在验证或测试集上的性能
    """
    model.model.eval()
    auc, r_acc, f_acc = evaluate(model, dataset_path, mode=mode)
    return {
        'auc': auc,
        'r_acc': r_acc,
        'f_acc': f_acc
    }

if __name__ == '__main__':
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化数据加载器（使用FaceForensicsLoader替代原有的FFDataset）
    transforms = get_transforms()
    
    # 训练集
    train_dataset = FaceForensicsLoader(
        root=dataset_path,
        split='train',
        frame_count=frame_num,
        transform=transforms['train']
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # 验证集
    val_dataset = FaceForensicsLoader(
        root=dataset_path,
        split='val',
        frame_count=frame_num,
        transform=transforms['val']
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    print(f"Train Dataset: {len(train_dataset)}, Val Dataset: {len(val_dataset)}")

    # 初始化检查点和日志
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    os.makedirs(ckpt_path, exist_ok=True)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val_auc = 0.
    ckpt_model_name = 'best.pkl'
    
    # 初始化模型
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    
    # 开始训练
    for epoch in range(max_epoch):
        logger.debug(f'Epoch {epoch+1}/{max_epoch}')
        
        # 更新采样策略
        train_dataset.update_sampling_strategy(epoch, max_epoch)
        val_dataset.update_sampling_strategy(epoch, max_epoch)
        
        # 训练
        train_metrics = train_epoch(model, train_loader, device, epoch)
        logger.debug(f'(Train @ epoch {epoch+1}) loss: {train_metrics["loss"]:.4f}, '
                    f'auc: {train_metrics["auc"]:.4f}, acc: {train_metrics["acc"]:.4f}')
        
        # 每10%的训练进度进行一次评估
        val_metrics = val_epoch(model, dataset_path, device, mode='valid')
        logger.debug(f'(Val @ epoch {epoch+1}) auc: {val_metrics["auc"]:.4f}, '
                    f'r_acc: {val_metrics["r_acc"]:.4f}, f_acc: {val_metrics["f_acc"]:.4f}')
        
        test_metrics = val_epoch(model, dataset_path, device, mode='test')
        logger.debug(f'(Test @ epoch {epoch+1}) auc: {test_metrics["auc"]:.4f}, '
                    f'r_acc: {test_metrics["r_acc"]:.4f}, f_acc: {test_metrics["f_acc"]:.4f}')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
            'best_val_auc': best_val_auc,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics
        }
        torch.save(checkpoint, os.path.join(ckpt_path, f'checkpoint_{epoch+1}.pkl'))
        logger.debug(f"Saved checkpoint for epoch {epoch+1}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.model.state_dict(), os.path.join(ckpt_path, 'best_model.pth'))
            logger.debug(f"New best model saved with AUC: {best_val_auc:.4f}")
    
    # 最终测试
    model.model.eval()
    final_metrics = val_epoch(model, dataset_path, device, mode='test')
    logger.debug(f'(Final Test) auc: {final_metrics["auc"]:.4f}, '
                f'r_acc: {final_metrics["r_acc"]:.4f}, f_acc: {final_metrics["f_acc"]:.4f}')