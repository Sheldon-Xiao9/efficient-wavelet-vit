import torch
from torch import nn
from einops import rearrange
from transformers import ViTModel # type: ignore

class TKS(nn.Module):
    """
    TKS - Temporal Keyframe Selection (TKS) module
    
    TKS 是一个基于Transformer注意力机制的关键帧选择模块，通过 DAMA 模块提取的融合特征来选择关键帧。
    """
    def __init__(self, dim=256, dama_dim=128, vit_model='google/vit-base-patch16-224', freeze_vit=True):
        super().__init__()
        
        # 加载预训练ViT
        self.vit = ViTModel.from_pretrained(vit_model)
        if dama_dim != 3:  # 假设DAMA输出非3通道，需调整ViT输入
            original_conv = self.vit.embeddings.patch_embeddings.projection
            self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
                dama_dim, original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride
            )
        
        # 冻结ViT参数
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # 时序处理层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.vit.config.hidden_size,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True
        )
        self.temporal_layer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        
        # 动态门控评分网络
        self.dama_fused = nn.Linear(dama_dim, self.vit.config.hidden_size)
        self.gate_net = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size * 2, self.vit.config.hidden_size),
            nn.Sigmoid()
        )
        self.score_net = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, dama_out, batch_size=16):
        """
        Input:
            x: [B, T, C, H, W]
            dama_out: [B, C]
        Output:
            scores: [B, T]
        """
        B, T, C, H, W = x.shape
        frames = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # 分批处理视频帧
        all_features = []
        num_batches = (B*T + batch_size - 1) // batch_size  # 向上取整
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, B*T)
            
            if end_idx <= start_idx:
                break
                
            # 提取当前批次的帧
            batch_frames = frames[start_idx:end_idx]
            
            # 使用ViT处理当前批次
            with torch.no_grad() if self.vit.training == False else torch.enable_grad():
                outputs = self.vit(pixel_values=batch_frames)
                batch_features = outputs.last_hidden_state[:, 0]  # 使用CLS token作为特征
            
            all_features.append(batch_features)
        
        # 合并所有批次的结果
        vit_features = torch.cat(all_features, dim=0)
        vit_features = rearrange(vit_features, '(b t) d -> b t d', b=B, t=T)
        
        # 时序处理
        if hasattr(self, 'temporal_layer') and self.temporal_layer is not None:
            temporal_features = self.temporal_layer(vit_features)
        
        # DAMA特征融合
        dama_features = self.dama_fused(dama_out).unsqueeze(1).repeat(1, T, 1)
        
        # 特征融合
        fused_features = torch.cat([vit_features, dama_features], dim=-1)
        
        # 评分网络
        gate = self.gate_net(fused_features, dim=-1)
        fused_features = gate * temporal_features + (1 - gate) * fused_features
        scores = self.score_net(fused_features).squeeze(-1)
        
        return scores