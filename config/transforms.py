import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from typing import List
from facenet_pytorch import MTCNN # type: ignore

seed = 42
random.seed(seed)
torch.manual_seed(seed)

class FaceAlignTransform:
    """
    人脸对齐转换器
    """
    def __init__(self, margin):
        self.margin = margin
        self.mtcnn = MTCNN(
            margin=margin,  # 确保整个面部上下文被捕获
            keep_all=False,
            min_face_size=40,
            post_process=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        
    def __call__(self, image):
        """
        检测人脸并返回居中的人脸图像
        
        :param image: 输入图像
        :type image: PIL.Image or np.ndarray
        :return: 返回人脸图像
        :rtype: PIL.Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = image
            
        # 获取图像尺寸
        width, height = image.size
        
        try:
            # 检测人脸
            boxes, _ = self.mtcnn.detect(image)
            
            if boxes is not None and len(boxes) > 0:
                # 选择最大的人脸
                box = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]
                
                # 计算人脸中心
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # 计算人脸的宽度和高度
                face_size = max(box[2]-box[0], box[3]-box[1])
                
                # 计算裁剪区域
                crop_size = face_size + self.margin * 2
                
                # 计算裁剪区域，确保人脸居中
                left = int(max(0, center_x - crop_size / 2))
                top = int(max(0, center_y - crop_size / 2))
                right = int(min(width, center_x + crop_size / 2))
                bottom = int(min(height, center_y + crop_size / 2))
                
                # 裁剪图像
                return image.crop((left, top, right, bottom))
        except Exception as e:
            print(f"Failed to detect face: {e}")
            
        # 如果没有检测到人脸，则返回中心裁剪的图像
        center_crop_size = min(width, height)
        left = (width - center_crop_size) // 2
        top = (height - center_crop_size) // 2
        right = left + center_crop_size
        bottom = top + center_crop_size
        
        return image.crop((left, top, right, bottom))

def get_transforms():
    """
    获取数据转换器
    
    :return: 返回训练、验证和测试数据加载器
    :rtype: dict
    """
    # 定义转换
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        FaceAlignTransform(margin=20),
        transforms.Resize(450),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        FaceAlignTransform(margin=20),
        transforms.Resize(450),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        FaceAlignTransform(margin=20),
        transforms.Resize(450),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform
    }