import torch
from torch import nn
from network.dama import DAMA
from network.tcm import TCM

class DeepfakeDetector(nn.Module):
    def __init__(self, in_channels=3, dama_dim=128, batch_size=16):
        """
        结合DAMA和TCM的深度伪造检测器
        
        :param in_channels: 输入视频帧的通道数
        :type in_channels: int
        :param dama_dim: DAMA的输入特征维度
        :type dama_dim: int
        :param fusion_type: 特征融合方式（'concat'/'add'）
        :type fusion_type: str
        """
        super().__init__()
        
        self.dama_dim = dama_dim
        self.batch_size = batch_size
        
        # DAMA模块 - 提取关键帧的空间特征与时频特征
        self.dama = DAMA(in_channels=in_channels, dim=dama_dim, batch_size=batch_size)
        
        # TCM模块 - 分析视频帧序列的时序一致性
        self.tcm = TCM(dama_dim=dama_dim)
        
        # 特征融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(dama_dim*2, 2),
            nn.Softmax(dim=1)
        )
            
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(dama_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        
    def forward(self, x):
        """
        前向传播
        """
        B, T, C, H, W = x.shape
        
        # 1. DAMA处理帧序列
        dama_feats = self.dama(x)
        
        # 2. TCM分析时序一致性
        tcm_outputs = self.tcm(x, dama_feats)
        tcm_consistency = tcm_outputs['consistency_score'] # [B]
        tcm_feats = tcm_outputs['tcm_features'] # [B, T, D]
        
        # 3. 特征融合
        gate = self.fusion_gate(torch.cat([dama_feats, tcm_feats], dim=-1))
        fused_feats = gate[:, 0].unsqueeze(-1) * dama_feats + gate[:, 1].unsqueeze(-1) * tcm_feats
        
        # 4. 分类
        logits = self.classifier(fused_feats)
        
        return {
            'logits': logits,
            'dama_feats': dama_feats,
            'tcm_consistency': tcm_consistency
        }