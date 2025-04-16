import torch
import yaml # type: ignore
from torch import nn
from torch.nn import functional as F
from network.dama import DAMA
from network.mwt import MWT
from network.sfe import EfficientViT

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
        self.ablation_config = ['dynamic', 'sfe_only', 'sfe_mwt']
        
        # 加载配置
        with open('config/architecture.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # DAMA模块
        self.dama = DAMA(in_channels=in_channels, dim=dama_dim, num_heads=4, levels=3, batch_size=batch_size)
        
        self.mwt = MWT(in_channels=in_channels, dama_dim=dama_dim)
        self.sfe = EfficientViT(
            config=self.config,
            channels=1280,
            feat_dim=dama_dim,
            selected_efficient_net=0
        )
        
        self.sfe_cls = EfficientViT(
            config=self.config,
            channels=1280,
            feat_dim=dama_dim,
            selected_efficient_net=0,
            output_mode='cls'
        )
        
        # 特征融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(dama_dim*2, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.feat_pooler = nn.AdaptiveAvgPool2d(1)
            
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(dama_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, batch_size, ablation):
        """
        前向传播
        """
        if batch_size is not None:
            self.batch_size = batch_size
        
        if ablation is not None:
            self.ablation = ablation
            
        B, K, C, H, W = x.size()
        
        # 根据消融配置选择特征
        if self.ablation == 'dynamic':
            # DAMA特征
            dama_feats = self.dama(x, batch_size=self.batch_size)
            
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
        elif self.ablation == 'sfe_only':
            all_logits = []
            
            for start_idx in range(0, K, self.batch_size):
                end_idx = min(start_idx + self.batch_size, K)
                batch_frames = x[:, start_idx:end_idx].flatten(0, 1)
                
                logits = self.sfe_cls(batch_frames)
                logits = logits.view(B, -1, 1)
                all_logits.append(logits)
                
            # 拼接所有帧的logits
            all_logits = torch.cat(all_logits, dim=1)
            final_logits = all_logits.mean(dim=1)
            
            return {
                'logits': final_logits,
                'model': 'sfe_only'
            }
        elif self.ablation == 'sfe_mwt':
            # 简单拼接融合
            sfe_features = []
            mwt_features = []
            
            for start_idx in range(0, K, self.batch_size):
                end_idx = min(start_idx + self.batch_size, K)
                batch_frames = x[:, start_idx:end_idx].flatten(0, 1)
                
                # SFE特征
                sfe_feats = self.sfe(batch_frames)
                sfe_feats = self.feat_pooler(sfe_feats).squeeze(-1).squeeze(-1)
                sfe_feats = sfe_feats.view(B, -1, self.dama_dim)
                sfe_features.append(sfe_feats)
                
                # MWT特征
                mwt_feats = self.mwt(batch_frames)
                mwt_feats = mwt_feats.squeeze(-1).squeeze(-1)
                mwt_feats = mwt_feats.view(B, -1, self.dama_dim)
                mwt_features.append(mwt_feats)
            
            # 拼接所有帧的特征
            sfe_features = torch.cat(sfe_features, dim=1).mean(dim=1)
            mwt_features = torch.cat(mwt_features, dim=1).mean(dim=1)
            
            # 特征融合
            combined = torch.cat([sfe_features, mwt_features], dim=1)
            gate_weights = self.fusion_gate(combined)
            gate_weights = F.softmax(gate_weights, dim=1)
            
            weighted_sfe = sfe_features * gate_weights[:, 0:1]
            weighted_mwt = mwt_features * gate_weights[:, 1:2]
            fused = weighted_sfe + weighted_mwt
            
            # 分类
            logits = self.classifier(fused)
            
            return {
                'logits': logits,
                'sfe': sfe_features,
                'mwt': mwt_features,
                'model': 'sfe_mwt'
            }
                
        
    def configure_ablation(self, ablation):
        """
        配置消融项
        """
        if ablation in self.ablation_config:
            self.ablation = ablation
        else:
            raise ValueError(f"Invalid ablation config: {ablation}.")