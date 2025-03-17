import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    二分类 Focal Loss 实现
        用于解决 FF++ 等数据集中数据不平衡问题
    
    :param alpha: 正类样本的平衡权重，值越大，正类样本的权重越大
    :type alpha: float, list
    :param gamma: 难易样本调节因子，值越大，损失就越关注难样本
    :type gamma: float
    :param reduction: 损失函数的计算方式，可选值有 "mean", "sum", "none"
    :type reduction: str
    """
    def __init__(self, alpha=0.75, gamma=2, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, float):
                self.alpha = torch.Tensor([alpha, 1-alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha
            self.alpha = self.alpha.to(torch.float32)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        logpt = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        pt = torch.exp(-logpt)
        focal_loss = self.alpha * (1-pt)**self.gamma * logpt
        
        # 计算损失
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss