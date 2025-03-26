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
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        """
        计算二分类 Focal Loss
        
        :param input: 模型输出的logits，形状为 (B, 1)，B 为 batch size，第二维度为正负类别
        :type input: torch.Tensor
        :param target: 真实标签，形状为 (B, 1)，B 为 batch size，第二维度为正负类别
        """
        p = torch.sigmoid(input)
        
        # 计算交叉熵项
        ce_loss = F.binary_cross_entropy(p, target, reduction="none")
        
        # 计算调制因子 (1 - p_t)^gamma
        p_t = p * target + (1 - p) * (1 - target)
        modulating_factor = (1 - p_t) ** self.gamma
        
        # 动态分配 alpha 权重
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # 总损失
        focal_loss = alpha_weight * modulating_factor * ce_loss
        
        # 计算损失
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss