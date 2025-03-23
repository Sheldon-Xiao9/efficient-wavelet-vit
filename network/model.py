import torch
# import copy
from torch import nn
from network.dama import DAMA
from network.mwt import MWT
from network.sfe import SFE

class DeepfakeDetector(nn.Module):
    def __init__(self, in_channels=3, dama_dim=128, batch_size=16, ablation='dynamic'):
        """
        结合DAMA和TCM的深度伪造检测器
        
        :param in_channels: 输入视频帧的通道数
        :type in_channels: int
        :param dama_dim: DAMA的输入特征维度
        :type dama_dim: int
        :param ablation: 实验割除项，用于消融实验
        :type ablation: str
        """
        super().__init__()
        
        self.dama_dim = dama_dim
        self.in_channels = in_channels
        self.batch_size = batch_size
        
        # 消融配置
        self.ablation_config = ['dynamic', 'space', 'freq']
        
        # DAMA模块
        self.dama = DAMA(in_channels=in_channels, dim=dama_dim, num_heads=4, levels=3, batch_size=batch_size)
        
        self.mwt = MWT(in_channels=in_channels, dama_dim=dama_dim)
        self.sfe = SFE(in_channels=in_channels)
        
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
        
        
    def forward(self, x, batch_size, ablation):
        """
        前向传播
        """
        if batch_size is not None:
            self.batch_size = batch_size
        
        if ablation is not None:
            self.ablation = ablation
            
        # DAMA特征
        dama_feats = self.dama(x, batch_size=self.batch_size)
        
        # 根据消融配置选择特征
        if self.ablation == 'dynamic':
            # 使用融合特征
            fused_feats = dama_feats['fused']
            space_feats = dama_feats['space']
            freq_feats = dama_feats['freq']
            
            logits = self.classifier(fused_feats)
            
            return {
                'logits': logits,
                'fused': fused_feats,
                'space': space_feats,
                'freq': freq_feats
            }
        elif self.ablation == 'space':
            # 使用空间特征
            space_feats = dama_feats['space']
            
            logits = self.classifier(space_feats)
            
            return {
                'logits': logits,
                'space': space_feats
            }
        elif self.ablation == 'freq':
            # 使用频域特征
            freq_feats = dama_feats['freq']
            
            logits = self.classifier(freq_feats)
            
            return {
                'logits': logits,
                'freq': freq_feats
            }
        else:
            # 使用融合特征
            fused_feats = dama_feats['fused']
            logits = self.classifier(fused_feats)
            
            return {
                'logits': logits,
                'fused': fused_feats
            }
        
    def configure_ablation(self, ablation):
        """
        配置消融项
        """
        if ablation in self.ablation_config:
            self.ablation = ablation
        else:
            raise ValueError(f"Invalid ablation config: {ablation}.")