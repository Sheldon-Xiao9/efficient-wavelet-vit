import torch
from torch import nn
from einops import rearrange
from transformers import ViTModel # type: ignore
from pytorch_wavelets import DWTForward # type: ignore

class TCM(nn.Module):
    """
    TCM - Temporal Consistency Module (TCM)
    
    基于ViT特征提取和轻量级时序分析的一致性检测模块
    """
    def __init__(self, dama_dim=128, vit_model='google/vit-base-patch16-224', freeze_vit=True):
        super().__init__()
        self.freeze_vit = freeze_vit
        
        # 加载预训练ViT
        self.vit = ViTModel.from_pretrained(vit_model)
        
        # 位置编码
        self.position_embedding = nn.Embedding(1000, self.vit.config.hidden_size)
        
        # 冻结ViT参数
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # 轻量级的GRU处理帧间关系
        self.gru = nn.GRU(
            input_size=self.vit.config.hidden_size,
            hidden_size=self.vit.config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 特征压缩层(双向GRU输出合并)
        self.feature_compressor = nn.Linear(self.vit.config.hidden_size * 2, self.vit.config.hidden_size)
        
        # DAMA特征处理
        self.dama_fused = nn.Linear(dama_dim, self.vit.config.hidden_size)
        
        # 一致性评分网络
        self.score_net = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 全局一致性特征提取器
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
        
        # ViT特征提取
        all_features = []
        for i in range(0, B*T, batch_size):
            batch_frames = frames[i:min(i+batch_size, B*T)]
            if self.freeze_vit:
                with torch.no_grad():
                    outputs = self.vit(pixel_values=batch_frames)
            else:
                outputs = self.vit(pixel_values=batch_frames)
            batch_feats = outputs.last_hidden_state[:, 0]
            all_features.append(batch_feats)
            torch.cuda.empty_cache()
        
        vit_features = torch.cat(all_features, dim=0)
        vit_features = vit_features.view(B, T, -1)  # [B, T, D]
        
        # 添加时序位置编码
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        vit_features = vit_features + self.position_embedding(position_ids)
        
        # GRU处理时序特征
        temporal_features, _ = self.gru(vit_features)  # [B, T, 2*D]
        temporal_features = self.feature_compressor(temporal_features)  # [B, T, D]
        
        # 计算帧间特征差异矩阵
        frame_features = temporal_features.unsqueeze(2)  # [B, T, 1, D]
        frame_features_t = temporal_features.unsqueeze(1)  # [B, 1, T, D]
        diff_matrix = torch.norm(frame_features - frame_features_t, dim=3)  # [B, T, T]
        
        # 每帧的不一致性得分(与其他帧的平均差异)
        inconsistency_scores = diff_matrix.mean(dim=2, keepdim=True)  # [B, T, 1]
        
        # DAMA特征与帧特征融合
        dama_global = self.dama_fused(dama_out).unsqueeze(1)  # [B, 1, D]
        dama_similarity = torch.cosine_similarity(
            temporal_features, dama_global.expand(-1, T, -1), dim=2
        ).unsqueeze(-1)  # [B, T, 1]
        
        # 融合不一致性得分和DAMA相似度
        combined_scores = inconsistency_scores * dama_similarity
        
        # 计算帧权重
        weights = torch.softmax(combined_scores.squeeze(-1), dim=1).unsqueeze(-1)  # [B, T, 1]
        
        # 全局一致性分数
        consistency_score = 1.0 - torch.mean(inconsistency_scores.squeeze(-1), dim=1)  # [B]
        
        # 加权特征聚合
        weighted_feats = (weights * temporal_features).sum(dim=1)  # [B, D]
        tcm_feats = self.global_consistency(weighted_feats)  # [B, dama_dim]
        
        return {
            'inconsistency_scores': inconsistency_scores,
            'tcm_features': tcm_feats,
            'frame_weights': weights.squeeze(-1)  # 返回每帧权重以便可视化
        }