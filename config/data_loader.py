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
                 methods=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter']):
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
        
        # 用于存储样本使用频率的字典
        self.all_fake_videos_by_method = {}
        self.video_usage_counts = {}
        self.current_epoch = 0
        
        # 加载视频位置
        self.real_videos, self.fake_videos = self._load_frames_dirs(self.methods)
        
        print(f"Loaded {len(self.real_videos)} real videos and {len(self.fake_videos)} fake videos")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.real_videos) + len(self.fake_videos)
    
    def _load_split(self):
        """加载数据集划分文件"""
        split_path = os.path.join(self.root, f'faceforensics-c23-processed/splits/splits/{self.split}.json')
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
        original_dir = os.path.join(self.root, f'faceforensics-c23-processed/ff/ff++/frames/original')
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
        
        # 确定每种方法需要提取的样本数量
        samples_per_method = len(real_dirs) // len(self.methods)
        if samples_per_method <= 0:
            raise ValueError(f"Invalid number of samples per method: {samples_per_method}")
        
        self.all_fake_videos_by_method = {method: [] for method in self.methods}
        
        # 收集每种伪造方法的所有可用视频
        method_videos = {}
        for method in self.methods:
            fake_dir = os.path.join(self.root, f'faceforensics-c23-processed/ff/ff++/frames/{method}')
            if not os.path.exists(fake_dir):
                raise FileNotFoundError(f"Fake videos directory '{fake_dir}' not found")
            
            for video_id in video_ids:
                target, source = video_id
                key = f"{target}_{source}"
                frames_dir = os.path.join(fake_dir, f'{target}_{source}')
                if os.path.exists(frames_dir):
                    if key not in method_videos:
                        method_videos[key] = []
                    method_videos[key].append({
                        'path': frames_dir,
                        'method': method,
                        'target': target,
                        'source': source,
                        'key': key
                    })
                
                # 记录视频使用次数
                self.all_fake_videos_by_method[method].append({
                    'path': frames_dir,
                    'method': method,
                    'target': target,
                    'source': source,
                    'key': key
                })
                self.video_usage_counts[f"{method}_{target}_{source}"] = 0
        
        if self.split == 'train':
            # 从每种方法中随机均匀提取样本
            fake_dirs = []
            method_counts = {method: 0 for method in self.methods}
            for video_id in list(method_videos.keys()):
                available_videos = method_videos[video_id]
                available_videos.sort(key=lambda x: method_counts[x['method']])
                selected_videos = available_videos[0]
                fake_dirs.append(selected_videos)
                method_counts[selected_videos['method']] += 1
                
                # 记录初始选择的视频使用次数加1
                key = f"{selected_videos['method']}_{selected_videos['key']}"
                self.video_usage_counts[key] += 1
            
            # 打乱伪造视频的顺序，确保不同方法的视频混合在一起
            random.shuffle(fake_dirs)
            
            print(f"Selected videos by method:")
            method_counts = {}
            for video in fake_dirs:
                method_counts[video['method']] = method_counts.get(video['method'], 0) + 1
            
            for method, count in method_counts.items():
                print(f"  - {method}: {count} videos")
        else:
            # 验证和测试阶段，使用所有伪造视频
            fake_dirs = []
            method_videos[method] = []
            for method in self.methods:
                fake_dir = os.path.join(self.root, f'faceforensics-c23-processed/ff/ff++/frames/{method}')
                for video_id in self.split_ids:
                    target, source = video_id
                    key = f"{target}_{source}"
                    frames_dir = os.path.join(fake_dir, key)
                    if os.path.exists(frames_dir):
                        method_videos[key].append({
                            'path': frames_dir,
                            'method': method,
                            'target': target,
                            'source': source,
                            'key': key
                        })
            for method, videos in method_videos.items():
                # 如果该方法下的视频不足，全部使用
                if len(videos) <= samples_per_method:
                    fake_dirs.extend(videos)
                else:
                    # 随机选择指定数量的样本
                    selected = random.sample(videos, samples_per_method)
                    fake_dirs.extend(selected)
            # 验证集保持固定顺序
            fake_dirs.sort(key=lambda x: x['key'])
        
        return real_dirs, fake_dirs
    
    def resample_fake_videos(self, epoch, max_epoch):
        """
        根据当前训练轮次重新抽样伪造视频
        
        :param epoch: 当前训练轮次
        :type epoch: int
        :param max_epoch: 最大训练轮次
        :type max_epoch: int
        """
        self.current_epoch = epoch
        samples_per_method = len(self.real_videos) // len(self.methods)
        
        early_stage = 0.4 # 前40%的训练轮次使用较为固定的样本
        mid_stage = 0.7 # 40%-70%的训练轮次逐渐增加样本多样性
        
        progress_ratio = min(1.0, epoch / (max_epoch * mid_stage))
        
        # 前期偏向使用固定样本，后期偏向使用新样本
        fixed_sample_ratio = max(0.0, 1.0 - progress_ratio)
        novelty_ratio = min(5.0, 1.0 + 4.0 * progress_ratio)
        
        print(f"  - Fixed sample ratio: {fixed_sample_ratio}")
        print(f"  - Novelty ratio: {novelty_ratio}")
        
        # 如果是早期阶段且固定样本比例高，则不更新样本
        if epoch < max_epoch * early_stage and fixed_sample_ratio > 0.8:
            print("  - Early stage, using fixed samples")
            return
        
        new_fake_dirs = []
        method_counts = {method: 0 for method in self.methods}
        
        # 重新抽样伪造视频
        for method in self.methods:
            available_videos = self.all_fake_videos_by_method[method]
            if not available_videos:
                continue
            
            # 根据使用频率为每个视频计算权重
            videos_weights = []
            for video in available_videos:
                video_key = f"{method}_{video['key']}"
                usage_count = self.video_usage_counts.get(video_key, 0)
                # 计算视频权重，使用次数为0的样本权重为1
                weight = 1.0 / (usage_count + 1) * (novelty_ratio if usage_count == 0 else 1.0)
                videos_weights.append((video, weight))
                
            # 根据权重随机选择样本
            weights = [w for _, w in videos_weights]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            selected_indices = np.random.choice(
                len(available_videos),
                size=samples_per_method,
                replace=True,
                p=probs
            )
            
            for idx in selected_indices:
                video = available_videos[idx]
                new_fake_dirs.append(video)
                method_counts[method] += 1
                
                # 更新视频使用次数
                key = f"{method}_{video['key']}"
                self.video_usage_counts[key] = self.video_usage_counts.get(key, 0) + 1
        
        # 打乱伪造视频的顺序，确保不同方法的视频混合在一起
        random.shuffle(new_fake_dirs)
        self.fake_videos = new_fake_dirs
        
        print(f"Resampled videos by method for epoch {epoch+1}:")
        for method, count in method_counts.items():
            print(f"  - {method}: {count} videos")
        
        # 输出未使用过的样本比例
        unused_count = sum(1 for k, v in self.video_usage_counts.items() if v == 0)
        total_count = len(self.video_usage_counts)
        print(f"Unused samples: {unused_count}/{total_count} ({unused_count/total_count*100:.2f}%)")
    
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
                blank = np.zeros((224, 224, 3), dtype=np.uint8)
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