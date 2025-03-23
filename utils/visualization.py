import os
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import roc_curve, precision_recall_curve # type: ignore
from typing import Dict, List

class EvalVisualization:
    """
    评估结果可视化
    """
    def __init__(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """
        绘制混淆矩阵
        
        :param conf_matrix: 混淆矩阵
        :type conf_matrix: np.ndarray
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'confusion_matrix.png'))
        plt.close()
        
    def plot_roc_curve(self, labels: np.ndarray, predictions: np.ndarray, auc_score:  float) -> None:
        """
        绘制ROC曲线
        
        :param labels: 真实标签
        :type labels: np.ndarray
        :param predictions: 预测概率
        :type predictions: np.ndarray
        :param auc_score: AUC得分
        :type auc_score: float
        """
        fpr, tpr, _ = roc_curve(labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'roc_curve.png'))
        plt.close()
        
    def plot_pr_curve(self, labels: np.ndarray, predictions: np.ndarray, ap_score: float) -> None:
        """
        绘制精确率-召回率曲线
        
        :param labels: 真实标签
        :type labels: np.ndarray
        :param predictions: 预测概率
        :type predictions: np.ndarray
        :param ap_score: AP得分
        :type ap_score: float
        """
        precision, recall, _ = precision_recall_curve(labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'pr_curve.png'))
        plt.close()
        
    def plot_pred_distribution(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        绘制预测分布
        
        :param predictions: 预测概率
        :type predictions: np.ndarray
        :param labels: 真实标签
        :type labels: np.ndarray
        """
        plt.figure(figsize=(8, 6))
        sns.kdeplot(predictions[labels == 0], label='Real', fill=True)
        sns.kdeplot(predictions[labels == 1], label='Fake', fill=True)
        plt.xlabel('Prediction')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'prediction_distribution.png'))
        plt.close()
        
    def plot_orth_vs_pred(self, orth_loss: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        绘制一致性得分与预测概率的关系
        
        :param orth_loss: 正交约束
        :type orth_loss: np.ndarray
        :param predictions: 预测概率
        :type predictions: np.ndarray
        :param labels: 真实标签
        :type labels: np.ndarray
        """
        if len(orth_loss) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(orth_loss[labels == 0], predictions[labels == 0], label='Real', alpha=0.7, c='blue')
        plt.scatter(orth_loss[labels == 1], predictions[labels == 1], label='Fake', alpha=0.7, c='red')
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Orth Loss Score')
        plt.ylabel('Prediction')
        plt.title('Orth Loss Scores vs Predictions')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_path, 'orth_vs_prediction.png'))
        plt.close()
    
    def plot_metrics(self, metrix, labels: np.ndarray, predictions: np.ndarray, orth_loss: np.ndarray) -> None:
        """生成所有评估指标的可视化"""
        self.plot_confusion_matrix(metrix['conf_matrix'])
        self.plot_roc_curve(labels, predictions, metrix['auc'])
        self.plot_pr_curve(labels, predictions, metrix['ap'])
        self.plot_pred_distribution(predictions, labels)
        
        if len(orth_loss) > 0:
            self.plot_orth_vs_pred(orth_loss, predictions, labels)
            
class TrainVisualization:
    """可视化训练过程"""
    def __init__(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        # 记录训练过程中的指标
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_orth_loss': [],
            'val_orth_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': [],
            'lr': [],
            'epochs': []
        }
        
    def update(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float) -> None:
        """
        更新训练过程中的指标
        
        :param epoch: 当前轮次
        :type epoch: int
        :param train_metrics: 训练集指标字典
        :type train_metrics: dict
        :param val_metrics: 验证集指标字典
        :type val_metrics: dict
        :param lr: 当前学习率
        :type lr: float
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        
        self.history['train_cls_loss'].append(train_metrics.get('cls_loss', 0))
        self.history['val_cls_loss'].append(val_metrics.get('cls_loss', 0))
        self.history['train_orth_loss'].append(train_metrics.get('orth_loss', 0))
        self.history['val_orth_loss'].append(val_metrics.get('orth_loss', 0))
        
        # 更新准确率和AUC
        self.history['train_acc'].append(train_metrics.get('acc', 0))
        self.history['val_acc'].append(val_metrics.get('acc', 0))
        self.history['train_auc'].append(train_metrics.get('auc', 0))
        self.history['val_auc'].append(val_metrics.get('auc', 0))
        
        # 如果学习率不为空，则记录学习率
        if lr is not None:
            self.history['lr'].append(lr)
    
    def _smooth_curve(self, values: list, weight: float = 0.7) -> list:
        """
        对数据进行平滑处理，使曲线更加平滑
        
        :param values: 待平滑的数据
        :type values: list
        :param weight: 平滑权重，默认为0.7
        :type weight: float
        :return: 平滑后的数据
        :rtype: list
        """
        smoothed_values = []
        last = values[0]
        for value in values:
            smoothed_val = last * weight + (1 - weight) * value
            smoothed_values.append(smoothed_val)
            last = smoothed_val
        return smoothed_values
            
    def plot_loss_curve(self, smoothing: bool = False) -> None:
        """
        绘制损失曲线
        
        :param smoothing: 是否进行平滑处理，默认为False
        :type smoothing: bool
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
        
        train_loss = self._smooth_curve(self.history['train_loss']) if smoothing else self.history['train_loss']
        val_loss = self._smooth_curve(self.history['val_loss']) if smoothing else self.history['val_loss']
        
        axes[0].plot(self.history['epochs'], train_loss, 'b-', linewidth=2, label='Train Loss')
        axes[0].plot(self.history['epochs'], val_loss, 'r-', linewidth=2, label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 一致性损失与分类损失
        train_cls = self._smooth_curve(self.history['train_cls_loss']) if smoothing else self.history['train_cls_loss']
        train_orth = self._smooth_curve(self.history['train_orth_loss']) if smoothing else self.history['train_orth_loss']
        val_cls = self._smooth_curve(self.history['val_cls_loss']) if smoothing else self.history['val_cls_loss']
        val_orth = self._smooth_curve(self.history['val_orth_loss']) if smoothing else self.history['val_orth_loss']
        
        axes[1].plot(self.history['epochs'], train_cls, 'c-', linewidth=2, label='Train Class. Loss')
        axes[1].plot(self.history['epochs'], train_orth, 'g-', linewidth=2, label='Train Orth. Loss')
        axes[1].plot(self.history['epochs'], val_cls, 'm-', linewidth=2, label='Val Class. Loss')
        axes[1].plot(self.history['epochs'], val_orth, 'y-', linewidth=2, label='Val Orth. Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Classification and Orthogonality Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'loss_curve.png'), dpi=300)
        plt.close()
        
    def plot_metrics_curve(self, smoothing: bool = False) -> None:
        """
        绘制指标曲线
        
        :param smoothing: 是否进行平滑处理，默认为False
        :type smoothing: bool
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True)
        
        # 准确率曲线
        train_acc = self._smooth_curve(self.history['train_acc']) if smoothing else self.history['train_acc']
        val_acc = self._smooth_curve(self.history['val_acc']) if smoothing else self.history['val_acc']
        
        axes[0].plot(self.history['epochs'], train_acc, 'g-', linewidth=2, label='Train Accuracy')
        axes[0].plot(self.history['epochs'], val_acc, 'y-', linewidth=2, label='Validation Accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training and Validation Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC曲线
        train_auc = self._smooth_curve(self.history['train_auc']) if smoothing else self.history['train_auc']
        val_auc = self._smooth_curve(self.history['val_auc']) if smoothing else self.history['val_auc']
        
        axes[1].plot(self.history['epochs'], train_auc, 'c-', linewidth=2, label='Train AUC')
        axes[1].plot(self.history['epochs'], val_auc, 'm-', linewidth=2, label='Validation AUC')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'metrics_curve.png'), dpi=300)
        plt.close()

    def plot_lr_curve(self) -> None:
        """绘制学习率曲线"""
        if not self.history['lr']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['epochs'], self.history['lr'], 'b-', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'lr_curve.png'))
        plt.close()

    def plot_combined_dashboard(self, smoothing: bool = False) -> None:
        """
        绘制训练过程中的所有曲线
        
        :param smoothing: 是否进行平滑处理，默认为False
        :type smoothing: bool
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 15), sharex=True)
        
        # 所有数据
        train_loss = self._smooth_curve(self.history['train_loss']) if smoothing else self.history['train_loss']
        val_loss = self._smooth_curve(self.history['val_loss']) if smoothing else self.history['val_loss']
        
        train_acc = self._smooth_curve(self.history['train_acc']) if smoothing else self.history['train_acc']
        val_acc = self._smooth_curve(self.history['val_acc']) if smoothing else self.history['val_acc']
        
        train_auc = self._smooth_curve(self.history['train_auc']) if smoothing else self.history['train_auc']
        val_auc = self._smooth_curve(self.history['val_auc']) if smoothing else self.history['val_auc']
        
        # 面板 1：损失曲线
        axes[0].plot(self.history['epochs'], train_loss, 'b-', linewidth=2, label='Train Loss')
        axes[0].plot(self.history['epochs'], val_loss, 'r-', linewidth=2, label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 面板 2：准确率曲线
        axes[1].plot(self.history['epochs'], train_acc, 'g-', linewidth=2, label='Train Accuracy')
        axes[1].plot(self.history['epochs'], val_acc, 'y-', linewidth=2, label='Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 面板 3：AUC曲线
        axes[2].plot(self.history['epochs'], train_auc, 'c-', linewidth=2, label='Train AUC')
        axes[2].plot(self.history['epochs'], val_auc, 'm-', linewidth=2, label='Validation AUC')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Training and Validation AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'combined_dashboard.png'), dpi=300)
        plt.close()
        
    def save_metrics(self) -> None:
        """保存训练过程中的指标"""
        try:
            data = {
                'epochs': self.history['epochs'],
                'train_loss': self.history['train_loss'],
                'val_loss': self.history['val_loss'],
                'train_cls_loss': self.history['train_cls_loss'],
                'val_cls_loss': self.history['val_cls_loss'],
                'train_orth_loss': self.history['train_orth_loss'],
                'val_orth_loss': self.history['val_orth_loss'],
                'train_acc': self.history['train_acc'],
                'val_acc': self.history['val_acc'],
                'train_auc': self.history['train_auc'],
                'val_auc': self.history['val_auc'],
                'lr': self.history['lr']
            }
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.output_path, 'training_history.csv'), index=False)
        except Exception as e:
            print(f"Error saving metrics: {e}")
        
    def plot_all(self, smoothing: bool = False) -> None:
        """
        绘制所有曲线
        
        :param smoothing: 是否进行平滑处理，默认为False
        :type smoothing: bool
        """
        self.plot_loss_curve(smoothing)
        self.plot_metrics_curve(smoothing)
        self.plot_lr_curve()
        self.plot_combined_dashboard(smoothing)
        self.save_metrics()
            