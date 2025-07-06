import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from network.model import DeepfakeDetector
from network.dama import DAMA
from network.dama import CrossAttention
from network.sfe import EfficientViT
from network.mwt import MWT

import yaml  # type: ignore
from config.transforms import get_transforms
from PIL import Image # type: ignore

# Grad-CAM实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x, batch_size=1, ablation='dynamic')
        logits = output['logits']

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        target = logits[:, class_idx]
        target.backward()

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError("Could not retrieve gradients or feature maps. Check hooks.")

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.detach().cpu().squeeze().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# 配置
MODEL_PATH = 'pretrained/xception-b5690688.pth'  # 替换为模型权重路径
ARCH_PATH = 'config/architecture.yaml' # SFE配置文件路径
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HOOK 缓存
feature_maps = {}
attention_weights = {}

def save_feature_map(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

def save_attention_weights(name):
    def hook(module, input, output):
        if hasattr(module, 'last_attn'):
            attention_weights[name] = module.last_attn.detach().cpu()
    return hook

# 交叉注意力的 Monkey Patch
def cross_attention_forward_with_save(self, x, context=None, kv_include_self=False):
    B, N, C = x.shape
    H = self.heads
    context = context if context is not None else x
    if kv_include_self:
        context = torch.cat((x, context), dim=1)
    q = self.to_q(x)
    k, v = self.to_kv(context).chunk(2, dim=-1)
    q = q.view(B, N, H, -1).transpose(1, 2)
    k = k.view(B, context.shape[1], H, -1).transpose(1, 2)
    v = v.view(B, context.shape[1], H, -1).transpose(1, 2)
    dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
    attn = self.attend(dots)
    self.last_attn = attn
    out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
    out = out.transpose(1, 2).contiguous().view(B, N, -1)
    return self.to_out(out)
setattr(CrossAttention, "forward", cross_attention_forward_with_save)

def show_feature_map_on_image(img, fmap, title, save_path):
    fmap = fmap.squeeze()
    if fmap.ndim == 3:
        fmap = fmap.mean(0)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
    fmap = cv2.resize(fmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(overlay[..., ::-1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def show_attention(attn, img, title, save_path):
    attn = attn.mean(1)[0]
    attn_map = attn.mean(0).cpu().numpy()
    plt.figure()
    plt.title(title + " (1D)")
    plt.plot(attn_map)
    plt.xlabel("Patch Index")
    plt.ylabel("Attention Weight")
    plt.savefig(save_path.replace('.png', '_1d.png'), bbox_inches='tight')
    plt.close()

def main(img_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms_dict = get_transforms()
    test_transform = transforms_dict['test']
    img_tensor = test_transform(img_rgb).unsqueeze(0).to(DEVICE)
    vis_img = img_tensor[0].cpu().clone()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vis_img = vis_img.permute(1,2,0).numpy() * std + mean
    vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)

    # --- 加载完整模型 ---
    model = DeepfakeDetector(batch_size=1, dama_dim=128)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    model.to(DEVICE)

    # --- Grad-CAM for Dynamic Mode ---
    print("--- Generating Grad-CAM for Dynamic Mode ---")
    # 目标层是DAMA模块内部的fusion_gate的第一个卷积层
    target_layer = model.dama.fusion_gate[0]
    grad_cam = GradCAM(model, target_layer)
    
    # 模拟视频输入 (B, K, C, H, W)
    video_tensor = img_tensor.unsqueeze(1) 
    
    # 计算 Grad-CAM
    cam_map = grad_cam(video_tensor, class_idx=0) # 假设类别0是'fake'
    grad_cam.remove_hooks()

    # 可视化 Grad-CAM
    grad_cam_save_path = os.path.join(save_dir, 'grad_cam_dynamic.png')
    show_feature_map_on_image(vis_img, torch.from_numpy(cam_map), 'Grad-CAM (Dynamic Mode)', grad_cam_save_path)
    print(f"Grad-CAM saved to {grad_cam_save_path}")


    # --- 原始特征图可视化 ---
    print("\n--- Generating Original Feature Maps ---")
    # 注册空间分支早期特征（EfficientNet-b0 features前几层）
    if hasattr(model.dama.sfe, "efficient_net"):
        print("[DEBUG] efficient_net children:", list(model.dama.sfe.efficient_net.named_children()))
        if hasattr(model.dama.sfe.efficient_net, "features"):
            model.dama.sfe.efficient_net.features[0].register_forward_hook(save_feature_map('early_space_f0'))
            model.dama.sfe.efficient_net.features[1].register_forward_hook(save_feature_map('early_space_f1'))
            model.dama.sfe.efficient_net.features[2].register_forward_hook(save_feature_map('early_space_f2'))
    # 注册频域分支早期特征（MWT第一层卷积）
    if hasattr(model.dama.mwt, "dwt"):
        # 只抓取wavelet_transform的高频输出
        orig_wavelet_transform = model.dama.mwt.wavelet_transform
        def wavelet_hook(x, target_size):
            ll, hf = orig_wavelet_transform(x, target_size)
            feature_maps['early_freq'] = hf.detach().cpu()
            return ll, hf
        model.dama.mwt.wavelet_transform = wavelet_hook

    model.dama.sfe.register_forward_hook(save_feature_map('space'))
    model.dama.mwt.register_forward_hook(save_feature_map('freq'))
    model.dama.fusion_gate.register_forward_hook(save_feature_map('fusion'))
    for m in model.dama.cross_att.layers:
        m[1].register_forward_hook(save_attention_weights('cross_attn_space2freq'))
        m[3].register_forward_hook(save_attention_weights('cross_attn_freq2space'))

    with torch.no_grad():
        # 使用DAMA的_process_frame来获取单帧特征
        _ = model.dama._process_frame(img_tensor)

    print("[DEBUG] space feature shape:", feature_maps['space'][0].shape)
    print("[DEBUG] freq feature shape:", feature_maps['freq'][0].shape)
    print("[DEBUG] fusion feature shape:", feature_maps['fusion'][0].shape)
    if 'early_space_f0' in feature_maps:
        print("[DEBUG] early_space_f0 feature shape:", feature_maps['early_space_f0'][0].shape)
    if 'early_space_f1' in feature_maps:
        print("[DEBUG] early_space_f1 feature shape:", feature_maps['early_space_f1'][0].shape)
    if 'early_space_f2' in feature_maps:
        print("[DEBUG] early_space_f2 feature shape:", feature_maps['early_space_f2'][0].shape)
    if 'early_freq' in feature_maps:
        print("[DEBUG] early_freq feature shape:", feature_maps['early_freq'][0].shape)

    show_feature_map_on_image(vis_img, feature_maps['space'][0], 'Space Feature', os.path.join(save_dir, 'space_feature.png'))
    show_feature_map_on_image(vis_img, feature_maps['freq'][0], 'Freq Feature', os.path.join(save_dir, 'freq_feature.png'))
    show_feature_map_on_image(vis_img, feature_maps['fusion'][0], 'Fusion Feature', os.path.join(save_dir, 'fusion_feature.png'))
    if 'early_space_f0' in feature_maps:
        show_feature_map_on_image(vis_img, feature_maps['early_space_f0'][0], 'EfficientNet-b0 features[0]', os.path.join(save_dir, 'efficientnet_b0_features0.png'))
    if 'early_space_f1' in feature_maps:
        show_feature_map_on_image(vis_img, feature_maps['early_space_f1'][0], 'EfficientNet-b0 features[1]', os.path.join(save_dir, 'efficientnet_b0_features1.png'))
    if 'early_space_f2' in feature_maps:
        show_feature_map_on_image(vis_img, feature_maps['early_space_f2'][0], 'EfficientNet-b0 features[2]', os.path.join(save_dir, 'efficientnet_b0_features2.png'))
    if 'early_freq' in feature_maps:
        show_feature_map_on_image(vis_img, feature_maps['early_freq'][0], 'MWT Wavelet HF', os.path.join(save_dir, 'mwt_wavelet_hf.png'))
    if 'cross_attn_space2freq' in attention_weights:
        show_attention(attention_weights['cross_attn_space2freq'], vis_img, 'Cross Attention (Space->Freq)', os.path.join(save_dir, 'cross_attention_space2freq.png'))
    if 'cross_attn_freq2space' in attention_weights:
        show_attention(attention_weights['cross_attn_freq2space'], vis_img, 'Cross Attention (Freq->Space)', os.path.join(save_dir, 'cross_attention_freq2space.png'))

    print(f'Visualization results saved to: {save_dir}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Input image path')
    parser.add_argument('--out', type=str, default='output/vis', help='Output directory for visualizations')
    args = parser.parse_args()
    main(args.img, args.out)
