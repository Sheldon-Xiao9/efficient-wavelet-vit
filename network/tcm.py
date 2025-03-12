import torch
from torch import nn
from einops import rearrange
from transformers import ViTModel # type: ignore

class TCM(nn.Module):
    """
    TCM - Temporal Consistency Module (TCM)
    
    TCM是一个基于ViT和Transformer的时序一致性分析模块，通过分析视频帧序列的时序特征
    并结合DAMA提取的全局特征，检测深度伪造视频中的时序不一致性。
    """
    def __init__(self, dama_dim=128, vit_model='google/vit-base-patch16-224', freeze_vit=True):
        super().__init__()
        self.freeze_vit = freeze_vit
        
        # 加载预训练ViT
        self.vit = ViTModel.from_pretrained(vit_model)
        
        # 位置编码
        self.position_embedding = nn.Embedding(1000, self.vit.config.hidden_size)
        # if dama_dim != 3:  # 假设DAMA输出非3通道，需调整ViT输入
        #     original_conv = self.vit.embeddings.patch_embeddings.projection
        #     self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
        #         dama_dim, original_conv.out_channels, 
        #         kernel_size=original_conv.kernel_size, 
        #         stride=original_conv.stride
        #     )
        
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
            nn.Linear(self.vit.config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.score_net = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.cross_att = nn.MultiheadAttention(
            embed_dim=self.vit.config.hidden_size,
            kdim=self.vit.config.hidden_size,
            vdim=self.vit.config.hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # 全局时序一致性特征提取器
        self.global_consistency = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, dama_dim)
        )
        
    def forward(self, x, dama_out, batch_size=16):
        """
        提取视频帧序列的时序特征并结合DAMA进行时序一致性分析
        """
        B, T, C, H, W = x.shape
        frames = rearrange(x, 'b t c h w -> (b t) c h w')
        
        all_features = []
        for i in range(0, B*T, batch_size):
            batch_frames = frames[i:min(i+batch_size, B*T)]
            if self.freeze_vit:
                with torch.no_grad():
                    outputs = self.vit(pixel_values=batch_frames)
            else:
                outputs = self.vit(pixel_values=batch_frames) # [B*T, D]
            batch_feats = outputs.last_hidden_state[:, 0] # [B*T, D]
            all_features.append(batch_feats)
            torch.cuda.empty_cache()
        vit_features = torch.cat(all_features, dim=0)
        vit_features = vit_features.view(B, T, -1) # [B, T, D]
        
        # 添加时序位置编码
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)  # [B, T]
        vit_features = vit_features + self.position_embedding(position_ids) # [B, T, D]
        
        # 时序处理
        temporal_features = self.temporal_layer(vit_features) # [B, T, D]
        
        # DAMA特征融合
        dama_features = self.dama_fused(dama_out).unsqueeze(1)
        
        # 交叉注意力
        att_output, _ = self.cross_att(
            query=temporal_features,
            key=dama_features,
            value=dama_features
        )
        # 残差连接
        fused_features = att_output + temporal_features
        
        # 评分网络
        gate = self.gate_net(fused_features) # [B, T, 1]
        gated_features = gate * temporal_features + (1 - gate) * fused_features
        
        # 帧得分
        scores = self.score_net(gated_features) # [B, T, 1]
        
        # 全局一致性分数计算
        temporal_diff = temporal_features - temporal_features.mean(dim=1, keepdim=True)
        consistency_score = torch.sqrt((temporal_diff ** 2).mean(dim=[1, 2]))  # [B]
        
        # 全局一致性特征提取
        weights = torch.softmax(scores.squeeze(-1), dim=1).unsqueeze(-1) # [B, T, 1]
        weighted_feats = (weights * temporal_features).sum(dim=1) # [B, D]
        tcm_feats = self.global_consistency(weighted_feats) # [B, dama_dim]
        
        return {
            'consistency_score': consistency_score,
            'tcm_features': tcm_feats
        }