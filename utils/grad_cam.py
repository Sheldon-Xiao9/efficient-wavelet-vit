import os
import torch
import numpy as np
import cv2
import yaml # type: ignore
from PIL import Image # type: ignore
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, CenterCrop # type: ignore

# 导入grad-cam库
from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore

from network.model import DeepfakeDetector


# ---- 配置 ----
MODEL_PATH = 'path/to/your/trained_model.pth' 
IMG_PATH = 'path/to/a/test_fake_image.png'
SAVE_DIR = 'output/vis_gradcam'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---- 数据预处理 (与你训练时保持一致) ----
def get_transforms():
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ---- 模型包装器 (为了让Grad-CAM能处理你的dynamic模型) ----
class DynamicModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(DynamicModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 调用模型的dynamic模式，并只返回logits
        # 注意：这里需要传入batch_size，如果你的模型内部需要的话
        outputs = self.model(x, batch_size=1, ablation='dynamic')
        return outputs['logits']


# ---- 主函数 ----
def main(model_path, img_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载和预处理图像
    rgb_img = Image.open(img_path).convert('RGB')
    transform = get_transforms()
    input_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    # 用于可视化的图像 (反归一化)
    vis_img = np.array(rgb_img) / 255.0
    vis_img = cv2.resize(vis_img, (224, 224))


    # 2. 创建两个独立的模型实例
    # sfe_only模型实例
    model_sfe_only = DeepfakeDetector(ablation='sfe_only').to(DEVICE)
    model_sfe_only.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model_sfe_only.eval()

    # dynamic模型实例
    model_dynamic_base = DeepfakeDetector(ablation='dynamic').to(DEVICE)
    model_dynamic_base.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model_dynamic_base.eval()
    # 用包装器包装dynamic模型
    model_dynamic = DynamicModelWrapper(model_dynamic_base)


    # 3. 定义Grad-CAM的目标层
    # SFE-only模型的目标层: EfficientNet-B0的最后一个特征块
    target_layer_sfe = [model_sfe_only.sfe_cls.efficient_net._blocks[-1]]

    # Dynamic模型的目标层: DAMA模块中fusion_gate的最后一层卷积
    # 这层直接产生了被池化前的二维融合特征图
    target_layer_dynamic = [model_dynamic_base.dama.fusion_gate[-1]]


    # 4. 定义Grad-CAM的目标输出 (我们关心分类logit)
    # 我们假设logit越大越代表'fake'
    targets = [ClassifierOutputTarget(0)]


    # 5. 生成并保存SFE-only的Grad-CAM图
    with GradCAM(model=model_sfe_only, target_layers=target_layer_sfe) as cam:
        grayscale_cam_sfe = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image_sfe = show_cam_on_image(vis_img, grayscale_cam_sfe, use_rgb=True)
        cv2.imwrite(os.path.join(save_dir, 'sfe_only_gradcam.png'), cam_image_sfe)
        print("SFE-only Grad-CAM saved.")


    # 6. 生成并保存Dynamic模型的Grad-CAM图
    with GradCAM(model=model_dynamic, target_layers=target_layer_dynamic) as cam:
        grayscale_cam_dynamic = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image_dynamic = show_cam_on_image(vis_img, grayscale_cam_dynamic, use_rgb=True)
        cv2.imwrite(os.path.join(save_dir, 'dynamic_model_gradcam.png'), cam_image_dynamic)
        print("Dynamic model Grad-CAM saved.")


    # 7. 将两张图拼接在一起进行对比
    img1 = cv2.imread(os.path.join(save_dir, 'sfe_only_gradcam.png'))
    img2 = cv2.imread(os.path.join(save_dir, 'dynamic_model_gradcam.png'))
    comparison_image = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(os.path.join(save_dir, 'comparison_gradcam.png'), comparison_image)
    print("Comparison image saved.")


if __name__ == '__main__':
    main(MODEL_PATH, IMG_PATH, SAVE_DIR)