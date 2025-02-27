import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # type: ignore
import random

class VideoLoader(Dataset):
    """
    VideoLoader - 加载视频数据集（FaceForensics++、DDFC等）
    
    """
    def __init__(self, root, split='train', frame_count=16, transform=None):
        super().__init__()