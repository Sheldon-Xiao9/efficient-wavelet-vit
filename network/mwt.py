import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward # type: ignore

class MWT(nn.Module):
    """
    MWT - Multi-level Wavelet Transformer
    
    MWT 是一个多级小波变换模块，用于提取视频帧的频域特征。
    """
    def __init__(self, in_channels=3, dama_dim=128, levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.dama_dim = dama_dim
        self.levels = levels
        
        # 初始化小波变换
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        
        # 频域卷积
        self.freq_conv = nn.Sequential(
            # nn.Conv2d(dama_dim, dama_dim, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(dama_dim),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(dama_dim, dama_dim, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(dama_dim),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(dama_dim, dama_dim, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(dama_dim),
            # nn.ReLU(inplace=True),
            nn.Conv2d(dama_dim, dama_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dama_dim),
            nn.ReLU(inplace=True)
        )
        
        self.freq_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dama_dim, dama_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(dama_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 高频通道压缩器（处理LH，HL,HH三个高频分量）
        self.hf_conv = nn.ModuleDict({
            'seperate': nn.ModuleList([
                nn.Sequential(
                # nn.Conv2d(in_channels, 3*in_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(3*in_channels),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(3*in_channels, 6*in_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(6*in_channels),
                # nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 6*in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(12*in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(3)]),
            'fusion': nn.Sequential(
                nn.Conv2d(18*in_channels, dama_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dama_dim),
                nn.ReLU(inplace=True)
            )
        })
        
        # 多尺度高频融合
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(levels*dama_dim, dama_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dama_dim),
            nn.ReLU(inplace=True)
        )
        
    def wavelet_transform(self, x, target_size):
        B, C, H, W = x.shape
        ll, hf = self.dwt(x)
        hf = hf[0].reshape(B, 3*C, H//2, W//2)
        
        if self.levels > 1:
            # 上采样
            hf = F.interpolate(hf, size=target_size, mode='bilinear')
        
        processed_hf = []
        for i in range(3):
            quantized_hf = self.hf_conv['seperate'][i](hf[:, i*C:(i+1)*C])
            processed_hf.append(quantized_hf)
        processed_hf = torch.cat(processed_hf, dim=1)
        hf_compressed = self.hf_conv['fusion'](processed_hf)
            
        return ll, hf_compressed
    
    def forward(self, x):
        """
        处理输入帧图像，提取频域特征
        
        :param x: 输入视频帧，形状为 [B, C, H, W]
        :type x: torch.Tensor
        :return: 返回频域特征，形状为 [B, D, H', W']
        :rtype: torch.Tensor
        """
        B, C, H, W = x.shape
        target_size = (H//2, W//2)
        
        current_input = x
        
        high_freqs = []
        for _ in range(self.levels):
            ll, hf = self.wavelet_transform(current_input, target_size)
            
            high_freqs.append(hf)
            current_input = ll
        
        multi_scale_feats = torch.cat(high_freqs, dim=1)
        fused_hf_feats = self.multiscale_fusion(multi_scale_feats)
        
        freq_feats = self.freq_conv(fused_hf_feats)
        freq_feats = self.freq_pool(freq_feats)
        
        return freq_feats

if __name__ == "__main__":
    # 测试MWT模块
    mwt = MWT()
    x = torch.randn(8, 3, 224, 224)
    y = mwt(x)
    print(y.shape)
    