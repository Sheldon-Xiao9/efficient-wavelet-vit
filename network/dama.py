import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d  # type: ignore
from pytorch_wavelets import DWTForward # type: ignore
from einops import rearrange

class DAMA(nn.Module):
    """
    DAMA - Dynamic Adaptive Multihead Attention (DAMA) module
    
    DAMA 是一个动态自适应多头注意力模块，用于提取关键帧的空间特征与时频特征。
    """
    def __init__(self, in_channels=3, dim=128, deform_groups=1, num_heads=4, levels=3):
        super().__init__()
        self.dim = dim
        self.deform_groups = deform_groups
        self.levels = levels
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
        
        # 频域分支输入（小波变换预输入）
        self.input_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        # 频域可变形卷积
        self.freq_deform_conv = DeformConv2d(dim, dim, kernel_size=3, padding=1, groups=deform_groups)
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
        ll, hf = self.dwt(x)
        high_freqs = []
        B, C, H, W = x.shape
        
        for level, hf in enumerate(hf):
            if level >= self.levels:
                break
            
            # 将高频分量拼接，并按通道数压缩（3*C -> C）
            hf = hf[0].reshape(B, 3*C, H//2, W//2)
            hf_feats = self.hf_conv(hf)
            high_freqs.append(hf_feats)
            
            ll, hf = self.dwt(ll)
            
        return torch.cat(high_freqs, dim=1)
        
    def forward(self, x, batch_size=16):
        # x: [B, K, C, H, W]
        B, K, C, H, W = x.shape
        all_feats = []
        
        # 分批处理视频帧
        for start_idx in range(0, K, batch_size):
            end_idx = min(start_idx + batch_size, K)
            current_batch_size = end_idx - start_idx
            
            frame_rgb = x[:, start_idx:end_idx].reshape(-1, C, H, W) # [B*K, C, H, W]
            frame = self.input_conv(frame_rgb)
            
            hf = self.wavelet_transform(frame)
            hf_feats = self.multiscale_fusion(hf)
            
            # 生成偏移量（融合原始帧与低频）
            hf_upsampled = F.interpolate(hf_feats, size=(H,W), mode='bilinear')
            offset_input = torch.cat([frame, hf_upsampled], dim=1)
            offsets = self.shared_offset_net(offset_input)
            
            space_offsets = offsets[:,:2*3*3*self.deform_groups]
            freq_offsets = offsets[:,2*3*3*self.deform_groups:]
            
            # 空间处理
            space_feats = self.space_deform_conv(frame, space_offsets)
            space_feats = self.space_att(space_feats) # [B, dim, H/2, W/2]
            
            # 频域处理
            freq_feats = self.freq_deform_conv(hf_upsampled, freq_offsets)
            freq_feats = self.freq_att(freq_feats) # [B, dim, H/2, W/2]
            
            # 动态门控
            gate_input = torch.cat([space_feats, freq_feats], dim=1)
            gate_weights = self.gate_net(gate_input) # [B, 2]
            
            # 交叉注意力融合
            B_batch = B * current_batch_size
            H_half, W_half = H // 2, W // 2
            
            space_feats = space_feats.view(B_batch, self.dim, -1).permute(0, 2, 1) # [B*K, H/2*W/2, dim]
            freq_feats = freq_feats.view(B_batch, self.dim, -1).permute(0, 2, 1) # [B*K, H/2*W/2, dim]
            
            # 交叉注意力
            fused_flat, _ = self.cross_att(
                query=space_feats,
                key=freq_feats,
                value=freq_feats
            )
            
            fused_feats = fused_flat.permute(0, 2, 1).view(B_batch, self.dim, H_half, W_half)
            
            # 动态门控加权融合
            weighted_fused = gate_weights[:,0].view(B_batch,1,1,1) * space_feats + gate_weights[:,1].view(B_batch,1,1,1) * fused_feats
            
            
            batch_results = weighted_fused.view(B, current_batch_size, self.dim, H_half, W_half) # [B, K, dim, H/2, W/2]
            all_feats.append(batch_results)
            
        # 连接所有批次的特征
        features = torch.cat(all_feats, dim=1) # [B, K, dim, H/2, W/2]
        return features.mean(dim=[1, 3, 4]) # [B, dim]
