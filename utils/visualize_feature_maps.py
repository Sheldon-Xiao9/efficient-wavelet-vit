import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from network.dama import DAMA
from network.dama import CrossAttention
from network.sfe import EfficientViT
from network.mwt import MWT

import yaml  # type: ignore
from config.transforms import get_transforms
from PIL import Image # type: ignore

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
        # CrossAttention 的 attn 在 forward 里是局部变量，需 monkey patch
        if hasattr(module, 'last_attn'):
            attention_weights[name] = module.last_attn.detach().cpu()
    return hook

# 交叉注意力的 Monkey Patch
def cross_attention_forward_with_save(self, x, context=None, kv_include_self=False):
    # x: [B, N, C]
    B, N, C = x.shape
    H = self.heads
    context = context if context is not None else x
    if kv_include_self:
        context = torch.cat((x, context), dim=1)
    q = self.to_q(x)
    k, v = self.to_kv(context).chunk(2, dim=-1)
    # [B, N, H*D] -> [B, H, N, D]
    q = q.view(B, N, H, -1).transpose(1, 2)
    k = k.view(B, context.shape[1], H, -1).transpose(1, 2)
    v = v.view(B, context.shape[1], H, -1).transpose(1, 2)
    dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
    attn = self.attend(dots)
    self.last_attn = attn  # 保存注意力权重
    out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
    out = out.transpose(1, 2).contiguous().view(B, N, -1)
    return self.to_out(out)
setattr(CrossAttention, "forward", cross_attention_forward_with_save)

# 可视化函数
def show_feature_map_on_image(img, fmap, title, save_path):
    fmap = fmap.squeeze()
    if fmap.ndim == 3:
        fmap = fmap.mean(0)  # [C, H, W] -> [H, W]
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
    # attn: [B, heads, N, N]，N=patch数
    attn = attn.mean(1)[0]  # 取第一个样本，所有 head 平均
    attn_map = attn.mean(0).numpy()  # [N]
    size = int(np.sqrt(attn_map.shape[0]))
    attn_map = attn_map.reshape(size, size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(overlay[..., ::-1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main(img_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # 读取图片
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 应用transform
    transforms_dict = get_transforms()
    test_transform = transforms_dict['test']
    # 兼容 ToPILImage，直接传 numpy 数组
    img_tensor = test_transform(img_rgb).unsqueeze(0).to(DEVICE)
    # 用于可视化的transform后帧
    vis_img = img_tensor[0].cpu().clone()
    # 逆归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vis_img = vis_img.permute(1,2,0).numpy() * std + mean
    vis_img = np.clip(vis_img * 255, 0, 255).astype(np.uint8)

    # 加载模型
    with open(ARCH_PATH, 'r') as f:
        config = yaml.safe_load(f)
    model = DAMA(in_channels=3, dim=128, num_heads=4, levels=3, batch_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    model.to(DEVICE)

    # 注册 hook
    # EfficientViT
    model.sfe.register_forward_hook(save_feature_map('space'))
    # MWT
    model.mwt.register_forward_hook(save_feature_map('freq'))
    # DAMA fusion_gate
    model.fusion_gate.register_forward_hook(save_feature_map('fusion'))
    # CrossAttention
    for m in model.cross_att.layers:
        m[1].register_forward_hook(save_attention_weights('cross_attn_space2freq'))
        m[3].register_forward_hook(save_attention_weights('cross_attn_freq2space'))

    # 推理
    with torch.no_grad():
        _ = model._process_frame(img_tensor)

    # 可视化
    show_feature_map_on_image(vis_img, feature_maps['space'][0], 'Space Feature', os.path.join(save_dir, 'space_feature.png'))
    show_feature_map_on_image(vis_img, feature_maps['freq'][0], 'Freq Feature', os.path.join(save_dir, 'freq_feature.png'))
    show_feature_map_on_image(vis_img, feature_maps['fusion'][0], 'Fusion Feature', os.path.join(save_dir, 'fusion_feature.png'))
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
