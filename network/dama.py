import os
import torch
import yaml # type: ignore
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from network.mwt import MWT
from network.sfe import EfficientViT

os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        
        context = context if context is not None else x
        
        if kv_include_self:
            context = torch.cat((x, context), dim=1)
            
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = self.attend(dots)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

# 双向交叉注意力变换器
class BidirectionalCrossTransformer(nn.Module):
    def __init__(self, dim, depth=1, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ]))

    def forward(self, space_tokens, freq_tokens):
        for space_norm, space_attend_freq, freq_norm, freq_attend_space in self.layers:
            # 空间特征关注频域特征
            space_tokens_norm = space_norm(space_tokens)
            space_tokens = space_tokens + space_attend_freq(space_tokens_norm, freq_tokens, kv_include_self=True)
            
            # 频域特征关注空间特征
            freq_tokens_norm = freq_norm(freq_tokens)
            freq_tokens = freq_tokens + freq_attend_space(freq_tokens_norm, space_tokens, kv_include_self=True)
            
        return space_tokens, freq_tokens

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
        self.sfe = EfficientViT(
            config=yaml.safe_load(open('config/architecture.yaml', 'r')),
            channels=1280,
            selected_efficient_net=1,
            feat_dim=dim,
            output_mode='feature_map'
        )
        
        # 频域分支初始化
        self.mwt = MWT(in_channels=in_channels, dama_dim=dim, levels=levels)
        
        # 动态门控网络
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2*dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim//2, 3), # 输出空间与频域的门控权重
            nn.Softmax(dim=1)
        )
        
        # 多头注意力
        self.cross_att = BidirectionalCrossTransformer(
            dim=dim,
            depth=2,
            heads=num_heads,
            dim_head=dim//num_heads,
            dropout=0.1
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def _process_frame(self, frame):
        B, C, H, W = frame.shape
        
        # 空间处理
        space_feats = self.sfe(frame)
        
        # 频域处理
        freq_feats = self.mwt(frame)
        
        # 保存原始特征形状
        H_out, W_out = space_feats.shape[-2:]
        
        # 双向交叉注意力融合
        space_flat = rearrange(space_feats, 'B C H W -> B (H W) C')
        freq_flat = rearrange(freq_feats, 'B C H W -> B (H W) C')
        space_enhanced, freq_enhanced = self.cross_att(space_flat, freq_flat)
        
        # 恢复特征形状
        space_feats = rearrange(space_enhanced, 'B (H W) C -> B C H W', H=H_out)
        freq_feats = rearrange(freq_enhanced, 'B (H W) C -> B C H W', H=H_out)
        
        # 特征拼接与融合
        concat_feats = torch.cat([space_feats, freq_feats], dim=1)
        fused_feats = self.fusion_gate(concat_feats)
        
        # 动态门控加权融合
        gate_input = torch.cat([space_feats, freq_feats], dim=1)
        gate_weights = self.gate_net(gate_input)
        
        weighted_fused = (
            gate_weights[:, 0].view(B,1,1,1) * space_feats +
            gate_weights[:, 1].view(B,1,1,1) * freq_feats +
            gate_weights[:, 2].view(B,1,1,1) * fused_feats
        )
        
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
            # torch.cuda.empty_cache()
            
            end_idx = min(start_idx + batch_size, K)
            batch_frames = x[:, start_idx:end_idx] # [B, batch_size, C, H, W]

            # 批处理帧
            features = self._process_frame(batch_frames.flatten(0,1))

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
