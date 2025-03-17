import os
import cv2
import glob
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class FaceForensicsLoader(Dataset):
    """
    FaceForensicsLoader - 加载视频数据集（FaceForensics++）
    """
    def __init__(self,
                 root,
                 split='train',
                 frame_count=24,
                 transform=None,
                 compression='C23',
                 methods=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']):
        """
        初始化FaceForensicsLoader
        
        :param root: FaceForensics++数据集的根目录
        :type root: str
        :param split: 数据集的划分（train/val/test）
        :type split: str
        :param frame_count: 每个视频的帧数
        :type frame_count: int
        :param transform: 数据预处理
        :type transform: callable
        :param compression: 视频压缩级别（'c0'=raw, 'c23'=hq, 'c40'=lq）
        :type compression: str
        :param methods: 伪造方法列表
        :type methods: list
        """
        super().__init__()
        self.root = root
        self.split = split
        self.frame_count = frame_count
        self.transform = transform
        self.compression = compression
        self.methods = methods
        
        # 加载数据集划分文件
        self.split_ids = self._load_split()
        
        # 加载视频位置
        self.real_videos, self.fake_videos = self._load_frames_dirs(self.methods)
        
        print(f"Loaded {len(self.real_videos)} real videos and {len(self.fake_videos)} fake videos")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.real_videos) + len(self.fake_videos)
    
    def _load_split(self):
        """加载数据集划分文件"""
        split_path = os.path.join(self.root, f'/home/python-projects/cnn-transformer/data/splits/{self.split}.json')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file '{split_path}' not found")
        
        with open(split_path, 'r') as f:
            return json.load(f)
        
    def _load_frames_dirs(self, method):
        """
        加载预载帧位置
        
        :return: 返回真实视频和伪造视频预载帧的路径``(real_videos, fake_videos)``
        :rtype: tuple
        
        ``real_videos``
            包含真实视频帧路径的列表
        ``fake_videos``
            包含伪造视频帧路径、方法和标签的字典列表
        """
        original_dir = os.path.join(self.root, f'ff++/frames/original')
        if not os.path.exists(original_dir):
            raise FileNotFoundError(f"Original video frames directory '{original_dir}' not found")
        
        # 获取当前分割的视频ID
        video_ids = []
        for pair in self.split_ids:
            video_ids.append(pair)
            
        # 收集原始视频路径
        real_dirs = []
        for video_id in video_ids:
            frames_dir = os.path.join(original_dir, f'{video_id[0]}')
            if os.path.exists(frames_dir):
                real_dirs.append(frames_dir)
            else:
                raise Exception(f"Original video '{frames_dir}' not found")
        
        # 收集伪造视频路径
        fake_dirs = []
        for method in self.methods:
            fake_dir = os.path.join(self.root, f'ff++/frames/{method}')
            if not os.path.exists(fake_dir):
                raise FileNotFoundError(f"Fake videos directory '{fake_dir}' not found")
            
            for video_id in video_ids:
                # 伪造视频帧的文件夹命名规则：<target>_<source>
                target, source = video_id
                frames_dir = os.path.join(fake_dir, f'{target}_{source}')
                if os.path.exists(frames_dir):
                    fake_dirs.append({
                        'path': frames_dir,
                        'method': method,
                        'target': target,
                        'source': source
                    })
                else:
                    raise Exception(f"Fake video '{frames_dir}' not found")
        
        return real_dirs, fake_dirs
    
    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        :param index: 数据索引
        :type index: int
        :return: 返回帧与标签的元组``(frames, label)``
        :rtype: tuple
        
        ``frames``
            帧序列张量 (T x C x H x W)
        ``label``
            标签，0=真实，1=伪造
        """
        if index < len(self.real_videos):
            # 真实视频
            frames_dir = self.real_videos[index]
            label = 0
        else:
            # 伪造视频
            fake_index = index - len(self.real_videos)
            if fake_index >= len(self.fake_videos):
                raise IndexError(f"Index '{index}' out of range")
            frames_dir = self.fake_videos[fake_index]['path']
            label = 1
        
        # 获取帧文件列表
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if not frame_files:
            frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        if not frame_files:
            raise ValueError(f"No frame images found in '{frames_dir}'")
        
        # 如果帧数超过需要的数量，选择均匀间隔的帧
        if len(frame_files) > self.frame_count:
            # 均匀选择frame_count个帧
            indices = np.linspace(0, len(frame_files) - 1, self.frame_count, dtype=int).tolist()
            selected_files = [frame_files[i] for i in indices]
        else:
            # 帧数不足，全部使用并可能重复最后一帧
            selected_files = frame_files
            # 如果帧数不足，重复最后一帧
            while len(selected_files) < self.frame_count:
                selected_files.append(frame_files[-1])
            
        # 读取帧
        frames = []
        for file_path in selected_files:
            img = cv2.imread(file_path)
            if img is not None: # 忽略无效帧
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            else:
                blank = np.zeros((256, 256, 3), dtype=np.uint8)
                frames.append(blank)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            
        # 转换为张量 (T x C x H x W)
        frames = torch.stack([frame for frame in frames if isinstance(frame, torch.Tensor)])
        
        return frames, label
    

class CelebDFLoader(Dataset):
    """
    Celeb DFLoader - 加载视频数据集（CelebDF v2）
    """