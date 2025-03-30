import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights # type: ignore
from einops import rearrange

class SFE(nn.Module):
    """
    SFE - Spatial Feature Extractor
    
    SFE 是一个空间特征提取器，使用EfficientNetV2-S提取空间特征，并经过Transformer Encoder处理获取输入DAMA的空间特征。
    """
    def __init__(self, in_channels=3, dim=128, num_heads=4):
        super().__init__()
        self.dim = dim
        
        # 加载预训练EfficientNetV2-S
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.efficient_backbone = efficientnet_v2_s(weights=weights)
        
        # 移除分类器
        self.efficient_backbone.classifier = nn.Identity()
        
        # 只使用前5个特征提取模块
        self.space_efficient = self.efficient_backbone.features[:5]
        DIM_FRONT_5 = 128
        
        for param in self.space_efficient.parameters():
            param.requires_grad = True
        
        # 空间卷积
        self.space_conv = nn.Conv2d(DIM_FRONT_5, dim, kernel_size=3, padding=1)
        self.space_pool = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1) # [B, dim, H, W]
        )
        
        # 线性映射
        self.embedding_linear = nn.Linear(dim, dim)
        
        # 位置编码
        self.register_buffer('positional_encodings', nn.Parameter(torch.zeros(1, 49, dim)), persistent=False)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=0.1,
            activation='gelu'
        )
        self.space_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # EfficientNetV2-S特征提取
        space_feats = self.space_efficient(x)
        space_feats = self.space_conv(space_feats)
        space_feats = self.space_pool(space_feats) # [B, dim, H/32, W/32]
        
        # 位置编码
        space_tokens = rearrange(space_feats, 'b c h w -> b (h w) c')
        seq_len = space_tokens.shape[1]
        pos_emb = self.positional_encodings[:, :seq_len, :]
        space_tokens += pos_emb
        
        # Transformer Encoder
        space_feats = self.space_transformer(space_tokens)
        
        # 调整形状
        space_feats = rearrange(space_feats, 'b (h w) c -> b c h w', h=H//32)
        
        return space_feats