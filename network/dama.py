import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights # type: ignore
from pytorch_wavelets import DWTForward # type: ignore
from einops import rearrange

class DAMA(nn.Module):
    """
    DAMA - Dynamic Adaptive Multihead Attention (DAMA) module
    
    DAMA 是一个动态自适应多头注意力模块，用于提取关键帧的空间特征与时频特征。
    """
    def __init__(self, in_channels=3, dim=128, num_heads=4, levels=3, batch_size=16):
        super().__init__()
        self.dim = dim
        self.levels = levels
        self.batch_size = batch_size
        # 初始化小波变换
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        
        # 加载预训练EfficientNetV2-S
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.efficient_backbone = efficientnet_v2_s(weights=weights)
        
        # 移除分类器
        self.efficient_backbone.classifier = nn.Identity()
        
        # 只使用前5个特征提取模块
        self.space_efficient = self.efficient_backbone.features[:5]
        DIM_FRONT_5 = 128
        
        # 冻结
        for param in self.space_efficient.parameters():
            param.requires_grad = False
        
        # 频域输入卷积
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # 空间卷积
        self.space_conv = nn.Conv2d(DIM_FRONT_5, dim, kernel_size=3, padding=1)
        self.space_pool = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1) # [B, dim, H, W]
        )
        
        # 频域卷积
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.freq_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # 高频通道压缩器（处理LH, HL, HH三个高频分量）
        self.hf_conv = nn.Sequential(
            nn.Conv2d(3*in_channels, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        # 多尺度高频融合
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(levels*dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
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
        
    def wavelet_transform(self, x):
        """
        Input: [B, C, H, W]
        Output: [B, 3*C, H/2, W/2]
        """
        B, C, H, W = x.shape
        high_freqs = []
        
        # 第一级小波变换
        ll, hf1 = self.dwt(x)
        hf1 = hf1[0].reshape(B, 3*C, H//2, W//2)
        hf1_compressed = self.hf_conv(hf1) # 提取特征同时压缩通道
        high_freqs.append(hf1_compressed)
        
        # 第二级小波变换
        if self.levels >=2:
            ll2, hf2 = self.dwt(ll)
            hf2 = hf2[0].reshape(B, 3*C, H//4, W//4)
            hf2_compressed = self.hf_conv(hf2)
            
            hf2_upsampled = F.interpolate(hf2_compressed, size=(H//2, W//2), mode='bilinear')
            high_freqs.append(hf2_upsampled)
        
        if self.levels >= 3:
            _, hf3 = self.dwt(ll2)
            hf3 = hf3[0].reshape(B, 3*C, H//8, W//8)
            hf3_compressed = self.hf_conv(hf3)
            
            hf3_upsampled = F.interpolate(hf3_compressed, size=(H//2, W//2), mode='bilinear')
            high_freqs.append(hf3_upsampled)
        
        return torch.cat(high_freqs, dim=1)
    
    def _process_frame(self, frame):
        B, C, H, W = frame.shape
        
        # 获取高频特征
        hf = self.wavelet_transform(frame)
        hf = self.multiscale_fusion(hf)
        
        # 生成偏移量（融合原始帧与低频）
        hf_upsampled = F.interpolate(hf, size=(H,W), mode='bilinear') # [B, dim, H, W]
        
        # 空间处理
        space_feats = self.space_efficient(frame)
        space_feats = self.space_conv(space_feats)
        space_feats = self.space_pool(space_feats)
        
        # 频域处理
        freq_feats = self.freq_conv(hf_upsampled)
        freq_feats = self.freq_pool(freq_feats)
        
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
        
        return weighted_fused.mean(dim=[2,3])
        
    def forward(self, x, batch_size=16):
        # x: [B, K, C, H, W]
        B, K, C, H, W = x.shape
        mean_features = torch.zeros(B, self.dim, device=x.device)
        
        # 分批处理视频帧
        for start_idx in range(0, K, batch_size):
            # 清理缓存
            torch.cuda.empty_cache()
            
            end_idx = min(start_idx + batch_size, K)
            batch_frames = x[:, start_idx:end_idx] # [B, batch_size, C, H, W]

            # 批处理帧
            for i in range(batch_frames.shape[1]):
                frame_tensor = batch_frames[:, i].detach().requires_grad_(True)
                mean_features += checkpoint(self._process_frame, frame_tensor, use_reentrant=False)
            torch.cuda.empty_cache()

        # 连接所有批次的特征
                # 时序聚合
        return mean_features / K
