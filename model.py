import torch
from torch import nn
from network.dama import DAMA
from network.tks import TKS

class DeepfakeDetector(nn.Module):
    def __init__(self, in_channels=3, dim=128, dama_dim=128, candidate_ratio=0.5):
        super().__init__()
        self.dama_dim = dama_dim
        
        # DAMA
        self.dama = DAMA(in_channels=in_channels, dim=dama_dim)
        
        # TKS
        self.tks = TKS(in_channels=in_channels, dim=dim)
        
        # Candidate Keyframe Selection
        self.candidate_ratio = candidate_ratio
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dama_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        """forward

        Args:
            x [B, T, C, H, W]: input frames
        """
        B, T, C, H, W = x.shape
        
        # Step 1: TKS Initial Keyframe Selection(Use random features instead of DAMA features)
        with torch.no_grad():
            random_features = torch.randn(B, self.dama_dim, device=x.device)
            initial_scores = self.tks(x, random_features) # [B, T]
        
        # Choose candidate keyframes based on the ratio
        candidate_num = max(1, int(T * self.candidate_ratio))
        _, candidate_indices = initial_scores.topk(candidate_num, dim=1) # [B, candidate_num]
        
        # Extract candidate keyframes
        candidate_frames = []
        for i in range(B):
            candidate_frames.append(x[i, candidate_indices[i]])
        candidate_frames = torch.stack(candidate_frames) # [B, candidate_num, C, H, W]
        
        # Step 2: DAMA Feature Extraction
        dama_feats = self.dama(candidate_frames) # [B, candidate_num]
        
        # Step 3: TKS Final Keyframe Selection
        final_scores = self.tks(x, dama_feats)
        
        # Choose the final keyframe
        k = min(3, T)
        _, final_indices = torch.topk(final_scores, k, dim=1) # [B, k]
        
        # Extract final keyframes
        final_frames = []
        for i in range(B):
            final_frames.append(x[i, final_indices[i]])
        final_frames = torch.stack(final_frames) # [B, k, C, H, W]
        
        # Step 4: Classifier
        # Use the final keyframes to extract features
        final_feats = self.dama(final_frames)
        
        # Classifier
        logits = self.classifier(final_feats)
        
        return {
            'logits': logits,
            'candidate_scores': initial_scores,
            'final_scores': final_scores,
            'final_indices': final_indices
        }