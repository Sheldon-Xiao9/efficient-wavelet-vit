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
        if alpha is not None:
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        """
        计算二分类 Focal Loss
        
        :param input: 模型输出的logits，形状为 (B, 2)，B 为 batch size，第二维度为正负类别
        :type input: torch.Tensor
        :param target: 真实标签，形状为 (B, 2)，B 为 batch size，第二维度为正负类别
        """
        probs = F.softmax(input, dim=1)
        
        # 计算真实类别的概率
        pt = torch.sum(target * probs, dim=1)
        
        logpt = -torch.log(pt + 1e-8)
        factor = (1 - pt) ** self.gamma
        if self.alpha is not None and self.alpha.device != input.device:
            # alpha[0]: 正类样本的权重
            # alpha[1]: 负类样本的权重
            alpha_weight = torch.sum(self.alpha.to(input.device) * target, dim=1)
            focal_loss = factor * alpha_weight * logpt
        else:
            focal_loss = factor * logpt
            print(f"alpha lost: {focal_loss}")
        
        # 计算损失
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss