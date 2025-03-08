import os
import cv2
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
                 frame_count=16,
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
        self.real_videos, self.fake_videos = self._load_video_paths(self.methods)
        
        print(f"{len(self.real_videos)}")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.real_videos) + len(self.fake_videos)
    
    def _load_split(self):
        """加载数据集划分文件"""
        split_path = os.path.join(self.root, f'cnn-transformer/data/splits/{self.split}.json')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file '{split_path}' not found")
        
        with open(split_path, 'r') as f:
            return json.load(f)
        
    def _load_video_paths(self, method):
        """
        加载视频位置
        
        :return: 返回真实视频和伪造视频的路径``(real_videos, fake_videos)``
        :rtype: tuple
        
        ``real_videos``
            包含真实视频路径的列表
        ``fake_videos``
            包含伪造视频路径、方法和标签的字典列表
        """
        original_dir = os.path.join(self.root, f'dataset/1/FaceForensics++_{self.compression}/original')
        if not os.path.exists(original_dir):
            raise FileNotFoundError(f"Original videos directory '{original_dir}' not found")
        
        # 获取当前分割的视频ID
        video_ids = []
        for pair in self.split_ids:
            video_ids.append(pair)
            
        # 收集原始视频路径
        real_videos = []
        for video_id in video_ids:
            video_path = os.path.join(original_dir, f'{video_id[1]}.mp4')
            if os.path.exists(video_path):
                real_videos.append(video_path)
            else:
                raise Exception(f"Original video '{video_path}' not found")
        
        # 收集伪造视频路径
        fake_videos = []
        for method in self.methods:
            fake_dir = os.path.join(self.root, f'dataset/1/FaceForensics++_{self.compression}/{method}')
            if not os.path.exists(fake_dir):
                raise FileNotFoundError(f"Fake videos directory '{fake_dir}' not found")
            
            for video_id in video_ids:
                # 伪造视频命名规则：<target>_<source>.mp4
                target, source = video_id
                video_path = os.path.join(fake_dir, f'{target}_{source}.mp4')
                if os.path.exists(video_path):
                    fake_videos.append({
                        'path': video_path,
                        'method': method,
                        'target': target,
                        'source': source
                    })
                else:
                    raise Exception(f"Fake video '{video_path}' not found")
        
        return real_videos, fake_videos
    
    def _sample_frames(self, video_path: str):
        """
        从视频中采样帧
        
        :param video_path: 视频路径
        :type video_path: str
        :return: 返回帧列表
        :rtype: list
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Video '{video_path}' cannot be opened")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Video '{video_path}' has no frames")
        
        # 均匀采样帧
        if frame_count < self.frame_count:
            # 如果视频帧数少于指定帧数，则重复最后一帧
            indices = list(range(frame_count))
            indices.extend([frame_count - 1] * (self.frame_count - frame_count))
        else:
            indices = np.linspace(0, frame_count - 1, self.frame_count, dtype=int).tolist()
            
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转 RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 读取失败时复制上一帧或创建空白帧
                if frames:
                    frames.append(frames[-1])
                else:
                    blank = np.zeros((256, 256, 3), dtype=np.uint8)
                    frames.append(blank)
        
        cap.release()
        return frames
    
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
            video_path = self.real_videos[index]
            label = 0
        else:
            # 伪造视频
            fake_index = index - len(self.real_videos)
            if fake_index >= len(self.fake_videos):
                raise IndexError(f"Index '{index}' out of range")
            video_path = self.fake_videos[fake_index]['path']
            label = 1
        
        # 采样帧
        frames = self._sample_frames(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            
        # 转换为张量 (T x C x H x W)
        frames = torch.stack([frame for frame in frames if isinstance(frame, torch.Tensor)])
        
        return frames, label
    

class CelebDFLoader(Dataset):
    """
    Celeb DFLoader - 加载视频数据集（CelebDF v2）
    """