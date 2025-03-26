import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from network.mwt import MWT
from network.sfe import SFE
class DAMA(nn.Module):
    """
    DAMA - Dynamic Adaptive Multihaed Attention Module (DAMA)
    
    DAMA 是一个动态自适应多分支注意力模块，根据已提取的视频帧空间特征、频域特征，结合注意力机制实现视频帧的特征融合。
    """
    def __init__(self, in_channels=3, dim=128, num_heads=4, levels=3, batch_size=16):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.batch_size = batch_size

        # 空间分支初始化
        self.sfe = SFE(in_channels=in_channels, dim=dim, num_heads=num_heads)
        
        # 频域分支初始化
        self.mwt = MWT(in_channels=in_channels, dama_dim=dim, levels=levels)
        
        # 动态门控网络
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2*dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 2), # 输出空间与频域的门控权重
            nn.Softmax(dim=1)
        )
        
        # 多头注意力
        self.cross_att = nn.MultiheadAttention(dim, num_heads)
    
    def _process_frame(self, frame):
        B, C, H, W = frame.shape
        
        # 空间处理
        space_feats = self.sfe(frame)
        
        # 频域处理
        freq_feats = self.mwt(frame)
        
        # 动态门控
        gate_input = torch.cat([space_feats, freq_feats], dim=1)
        gate_weights = self.gate_net(gate_input)
        
        # 交叉注意力融合
        space_flat = rearrange(space_feats, 'B C H W -> B (H W) C')
        freq_flat = rearrange(freq_feats, 'B C H W -> B (H W) C')
        fused_feats, _ = self.cross_att(space_flat, freq_flat, freq_flat, key_padding_mask=None)
        fused_feats = rearrange(fused_feats, 'B (H W) C -> B C H W', H=H//32)
        
        # 动态门控加权融合
        weighted_fused = gate_weights[:,0].view(B,1,1,1) * space_feats + gate_weights[:,1].view(B,1,1,1) * fused_feats
        
        return {
            'fused': weighted_fused.mean(dim=[2,3]),
            'space': space_feats.mean(dim=[2,3]),
            'freq': freq_feats.mean(dim=[2,3])
        }
        
    def forward(self, x, batch_size=16):
        # x: [B, K, C, H, W]
        B, K, C, H, W = x.shape
        mean_fused = torch.zeros(B, self.dim, device=x.device)
        mean_space = torch.zeros(B, self.dim, device=x.device)
        mean_freq = torch.zeros(B, self.dim, device=x.device)
        
        # 分批处理视频帧
        for start_idx in range(0, K, batch_size):
            # 清理缓存
            torch.cuda.empty_cache()
            
            end_idx = min(start_idx + batch_size, K)
            batch_frames = x[:, start_idx:end_idx] # [B, batch_size, C, H, W]

            # 批处理帧
            features = self._process_frame(batch_frames.flatten(0,1))
            torch.cuda.empty_cache()
            
            features_fused = features['fused'].view(B, -1, self.dim)
            mean_fused += features_fused.sum(dim=1)
            
            features_space = features['space'].view(B, -1, self.dim)
            mean_space += features_space.sum(dim=1)
            
            features_freq = features['freq'].view(B, -1, self.dim)
            mean_freq += features_freq.sum(dim=1)
            
        mean_fused /= K # 连接所有批次的特征
        mean_space /= K
        mean_freq /= K

        # 连接所有批次的特征
        return {
            'fused': mean_fused,
            'space': mean_space,
            'freq': mean_freq
        }
