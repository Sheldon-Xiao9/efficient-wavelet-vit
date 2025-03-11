import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import DeformConv2d  # type: ignore
from pytorch_wavelets import DWTForward # type: ignore
from einops import rearrange

class DAMA(nn.Module):
    """
    DAMA - Dynamic Adaptive Multihead Attention (DAMA) module
    
    DAMA 是一个动态自适应多头注意力模块，用于提取关键帧的空间特征与时频特征。
    """
    def __init__(self, in_channels=3, dim=128, deform_groups=1, num_heads=4, levels=3, batch_size=16):
        super().__init__()
        self.dim = dim
        self.deform_groups = deform_groups
        self.levels = levels
        self.batch_size = batch_size
        # 初始化小波变换
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        
        # 统一偏移网络(输入=原始帧+上采样后的低频特征)
        self.shared_offset_net = nn.Sequential(
            nn.Conv2d(2 * dim, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 2*2*3*3*deform_groups, kernel_size=3, padding=1)
        )
        
        # 空间分支输入
        # 可变形卷积
        self.space_deform_conv = DeformConv2d(dim, dim, kernel_size=3, padding=1, groups=deform_groups)
        self.space_att = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1) # [B, dim, H, W]
        )
        
        # 频域分支输入（预输入）
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        # 频域卷积
        self.freq_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=deform_groups)
        self.freq_att = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1) # [B, dim, H, W]
        )
        # 高频通道压缩器（处理LH, HL, HH三个高频分量）
        self.hf_conv = nn.Sequential(
            nn.Conv2d(3*dim, dim, kernel_size=3, padding=1),
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
    
    def _process_frame(self, frame_rgb):
        B, C, H, W = frame_rgb.shape
        frame = self.input_conv(frame_rgb)
        
        # 获取高频特征
        hf = self.wavelet_transform(frame)
        hf = self.multiscale_fusion(hf)
        
        # 生成偏移量（融合原始帧与低频）
        hf_upsampled = F.interpolate(hf, size=(H,W), mode='bilinear')
        offset_input = torch.cat([frame, hf_upsampled], dim=1)
        offsets = self.shared_offset_net(offset_input)
        
        # 空间处理
        space_feats = self.space_deform_conv(frame, offsets)
        space_feats = self.space_att(space_feats)
        
        # 频域处理
        freq_feats = self.freq_conv(hf_upsampled)
        freq_feats = self.freq_att(freq_feats)
        
        # 动态门控
        gate_input = torch.cat([space_feats, freq_feats], dim=1)
        gate_weights = self.gate_net(gate_input)
        
        # 交叉注意力融合
        space_flat = rearrange(space_feats, 'B C H W -> B (H W) C')
        freq_flat = rearrange(freq_feats, 'B C H W -> B (H W) C')
        fused_feats, _ = self.cross_att(space_flat, freq_flat, freq_flat, key_padding_mask=None)
        fused_feats = rearrange(fused_feats, 'B (H W) C -> B C H W', H=H//2)
        
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

            for i in range(end_idx - start_idx):
                frame_rgb = batch_frames[:, i] # [B, C, H, W]
                
                frame_feats = self._process_frame(frame_rgb)
                mean_features += frame_feats
                
                torch.cuda.empty_cache()

        # 连接所有批次的特征
                # 时序聚合
        return mean_features / K
