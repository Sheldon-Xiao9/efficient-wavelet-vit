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
                 methods=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter'],
                 fixed_sample_ratio=1.0,
                 novelty_ratio=0.0,
                 single_method=None
                 ):
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
        self.fixed_sample_ratio = fixed_sample_ratio
        self.novelty_ratio = novelty_ratio
        self.single_method = single_method
        self.current_epoch = 0
        
        # 加载数据集划分文件
        self.split_ids = self._load_split()
        
        # 用于存储样本使用频率的字典
        self.all_fake_videos_by_method = {}
        self.video_usage_counts = {}
        
        # 加载视频位置
        self.real_videos, self.fake_videos = self._load_frames_dirs(self.methods)
        
        self._init_sampling_strategy() # 初始化样本采样策略
        
        print(f"Loaded {len(self.real_videos)} real videos and {len(self.fake_videos)} fake videos")
        
    def __len__(self):
        """返回数据集大小"""
        if self.split == 'train' or self.split == 'val':
            return len(self.real_videos) + len(self.current_fake)
        else:
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
        
        # 收集每种伪造方法的所有可用视频
        method_videos = {}
        for method in self.methods:
            fake_dir = os.path.join(self.root, f'faceforensics-c23-processed/ff/ff++/frames/{method}')
            if not os.path.exists(fake_dir):
                raise FileNotFoundError(f"Fake videos directory '{fake_dir}' not found")
            
            for video_id in video_ids:
                target, source = video_id
                key = f'{target}_{source}'
                frames_dir = os.path.join(fake_dir, f'{target}_{source}')
                if os.path.exists(frames_dir):
                    if key not in method_videos:
                        method_videos[key] = []
                    method_videos[key].append({
                        'path': frames_dir,
                        'method': method,
                        'target': target,
                        'source': source
                    })
        
        if self.split == 'test' and self.single_method is not None:
            # 测试集：收集所有该方法的伪造视频
            fake_dirs = []
            for video_id, methods_available in method_videos.items():
                for video in methods_available:
                    if video['method'] == self.single_method:
                        fake_dirs.append(video)
        else:
            # 训练集/验证集：从每种方法中随机均匀提取样本
            fake_dirs = []
            method_counts = {method: 0 for method in self.methods}
            for video_id, methods_available in method_videos.items():
                # 优先选择样本量少的方法
                methods_available.sort(key=lambda x: method_counts[x['method']])
                selected = methods_available[0]
                fake_dirs.append(selected)
                method_counts[selected['method']] += 1
            
        
        # 打乱伪造视频的顺序，确保不同方法的视频混合在一起
        random.shuffle(fake_dirs)
        
        print(f"Selected videos by method:")
        method_counts = {}
        for video in fake_dirs:
            method_counts[video['method']] = method_counts.get(video['method'], 0) + 1
        
        for method, count in method_counts.items():
            print(f"  - {method}: {count} videos")
        
        return real_dirs, fake_dirs
    
    def _init_sampling_strategy(self):
        """初始化样本采样策略，为训练和验证创建不同的样本集"""
        # 初始化视频使用计数
        for video in self.fake_videos:
            self.video_usage_counts[video['path']] = 0
        
        if self.split == 'train':
            # 训练集：初始固定样本集
            self.fixed_fake = random.sample(self.fake_videos, int(len(self.fake_videos) * self.fixed_sample_ratio))
            
            self.pool_fake = [v for v in self.fake_videos if v not in self.fixed_fake]
            
            self.current_fake = self.fixed_fake.copy()
        elif self.split == 'val':
            # 验证集：80%固定样本集，20%随机样本集
            random.seed(42)
            self.core_fake = random.sample(self.fake_videos, int(len(self.fake_videos) * 0.8))
            
            self.dynamic_pool_fake = [v for v in self.fake_videos if v not in self.core_fake]
            
            random.seed(42)
            self.dynamic_fake = random.sample(self.dynamic_pool_fake, min(int(len(self.fake_videos) * 0.2), len(self.dynamic_pool_fake)))
            
            self.current_fake = self.core_fake + self.dynamic_fake
    
    def _refresh_training_samples(self):
        """根据当前策略参数刷新训练样本"""
        num_fixed_fake = int(len(self.fake_videos) * self.fixed_sample_ratio)
        
        # 计算当前固定样本集的数量
        selected_fixed_fake = []
        if num_fixed_fake > 0:
            selected_fixed_fake = random.sample(self.fixed_fake, num_fixed_fake)
        
        # 剩余伪造视频数量
        remaining_fake = len(self.fake_videos) - num_fixed_fake
        
        # 按照使用频率排序伪造视频池
        self.pool_fake.sort(key=lambda x: self.video_usage_counts[x['path']])
        
        # 计算动态样本集的数量
        num_new_fake = int(remaining_fake * self.novelty_ratio)
        
        # 随机选择新的伪造视频
        num_random_fake = remaining_fake - num_new_fake
        # 防止pool_fake为空时报错
        if num_random_fake > 0 and len(self.pool_fake) > num_new_fake:
            random_samples = random.sample(self.pool_fake[num_new_fake:], min(num_random_fake, len(self.pool_fake) - num_new_fake))
        else:
            random_samples = []
        
        self.current_fake = (selected_fixed_fake + self.pool_fake[:num_new_fake] + random_samples)
        
        # 确保没有重复
        self.current_fake = list({v['path']: v for v in self.current_fake}.values())
        
        random.shuffle(self.current_fake)
    
    def update_sampling_strategy(self, epoch, max_epochs):
        """
        更新样本采样策略
        
        :param epoch: 当前训练轮次
        :type epoch: int
        :param max_epochs: 最大训练轮次
        :type max_epochs: int
        """
        self.current_epoch = epoch
        
        if self.split == 'train':
            early_stage = 0.3
            late_stage = 0.7
            
            # 如果是训练前期（30%），使用固定样本集
            if epoch < max_epochs * early_stage:
                self.fixed_sample_ratio = 1.0
                self.novelty_ratio = 0.0
                
                print(f"  - Fixed sample ratio: {self.fixed_sample_ratio:.2f}")
                print(f"  - Novelty ratio: {self.novelty_ratio:.2f}")
                print("  - Using fixed sample strategy")
            else:
                relative_epoch = epoch - (max_epochs * early_stage)
                transition_epochs = max_epochs * (late_stage - early_stage)
                progress_ratio = min(1.0, relative_epoch / transition_epochs)
                self.fixed_sample_ratio = max(0.3, 1.0 - progress_ratio)
                self.novelty_ratio = min(0.8, progress_ratio)
                
                print(f"  - Fixed sample ratio: {self.fixed_sample_ratio:.2f}")
                print(f"  - Novelty ratio: {self.novelty_ratio:.2f}")
            
            self._refresh_training_samples()
        elif self.split == 'val':
            # 验证集更新20%
            random.seed(42 + self.current_epoch)
            self.dynamic_fake = random.sample(self.dynamic_pool_fake, min(int(len(self.fake_videos) * 0.2), len(self.dynamic_pool_fake)))
            
            self.current_fake = self.core_fake + self.dynamic_fake
    
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
            if self.split == 'train' or self.split == 'val':
                if fake_index >= len(self.current_fake):
                    raise IndexError(f"Index '{index}' out of range")
                frames_dir = self.current_fake[fake_index]['path']
                
                # 更新视频使用计数
                self.video_usage_counts[frames_dir] = self.video_usage_counts.get(frames_dir, 0) + 1
            else:
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
    def __init__(self,
                 root,
                 split=['train', 'test'],
                 frame_count=24,
                 transform=None,
                 testing_file=None):
        """
        初始化CelebDFLoader
        :param root: CelebDF数据集的根目录
        :type root: str
        :param split: 数据集的划分（train/test）
        :type split: list
        :param frame_count: 每个视频的帧数
        :type frame_count: int
        :param transform: 数据预处理
        :type transform: callable
        :param testing_file: 测试集划分文件路径
        :type testing_file: str
        """
        super().__init__()
        self.root = root
        self.split = split
        self.frame_count = frame_count
        self.transform = transform
        self.testing_file = testing_file
        
        self.real_videos, self.synthetic_videos = self._load_frames_dirs()
        
        print(f"Loaded {len(self.real_videos)} real videos and {len(self.synthetic_videos)} synthetic videos")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.real_videos) + len(self.synthetic_videos)
    
    def _load_split(self):
        """
        加载测试集划分文件，不在该范围内的为训练集
        
        :return: 返回真实与伪造视频ID字典（仅测试集）
        :rtype: dict
        """
        if not os.path.exists(self.testing_file):
            raise FileNotFoundError(f"Testing file '{self.testing_file}' not found")
        
        test_videos = {'real': [], 'fake': []}
        skipped_youtube = 0
        
        with open(self.testing_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                label, video_path = parts[0], parts[1]
                
                # 跳过YouTube视频
                if 'youtube' in video_path.lower():
                    skipped_youtube += 1
                    continue
                
                video_id = video_path.split('/')[-1].split('.')[0]  # 提取视频ID
                
                if label == '1' and 'celeb-real' in video_path.lower():
                    test_videos['real'].append(video_id)
                elif label == '0' and 'celeb-synthesis' in video_path.lower():
                    test_videos['fake'].append(video_id)
        
        print(f"Skipped {skipped_youtube} YouTube videos")
        return test_videos
    
    def _load_frames_dirs(self):
        """
        加载预载帧位置
        
        :return: 返回真实视频和伪造视频预载帧的路径``(real_videos, synthetic_videos)``
        :rtype: tuple
        
        ``real_videos``
            包含真实视频帧路径的列表
        ``synthetic_videos``
            包含伪造视频帧路径、方法和标签的字典列表
        """
        real_dir = os.path.join(self.root, 'celebdf/frames/Celeb-real')
        synth_dir = os.path.join(self.root, 'celebdf/frames/Celeb-synthesis')
        
        if not os.path.exists(real_dir):
            raise FileNotFoundError(f"Real videos frames directory '{real_dir}' not found")
        if not os.path.exists(synth_dir):
            raise FileNotFoundError(f"Synthetic videos frames directory '{synth_dir}' not found")
        
        # 加载所有可用的视频
        all_real_videos = []
        for video_id in os.listdir(real_dir):
            frames_dir = os.path.join(real_dir, video_id)
            if os.path.isdir(frames_dir):
                all_real_videos.append((video_id, frames_dir))
        
        all_synthetic_videos = []
        for video_id in os.listdir(synth_dir):
            frames_dir = os.path.join(synth_dir, video_id)
            if os.path.isdir(frames_dir):
                all_synthetic_videos.append((video_id, frames_dir))
                
        # 加载测试集划分
        test_videos = self._load_split() if self.testing_file else {'real': [], 'fake': []}
        
        # 基于split参数选择视频
        real_videos = []
        synthetic_videos = []
        
        if 'test' in self.split:
            # 测试集
            for video_id, path in all_real_videos:
                if video_id in test_videos['real']:
                    real_videos.append(path)
                    
            for video_id, path in all_synthetic_videos:
                if video_id in test_videos['fake']:
                    synthetic_videos.append(path)
        else:
            # 训练集
            for video_id, path in all_real_videos:
                if video_id not in test_videos['real'] and path not in real_videos:
                    real_videos.append(path)
            
            for video_id, path in all_synthetic_videos:
                if video_id not in test_videos['fake'] and path not in synthetic_videos:
                    synthetic_videos.append(path)
                    
        return real_videos, synthetic_videos
    
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
            if fake_index >= len(self.synthetic_videos):
                raise IndexError(f"Index '{index}' out of range")
            frames_dir = self.synthetic_videos[fake_index]
            label = 1
        
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
        if not frame_files:
            raise FileNotFoundError(f"No frames found in '{frames_dir}'")
        
        # 如果帧数超过需要的数量，选择均匀间隔的帧
        if len(frame_files) > self.frame_count:
            indices = np.linspace(0, len(frame_files) - 1, self.frame_count, dtype=int).tolist()
            selected_files = [frame_files[i] for i in indices]
        else:
            # 帧数不足，全部使用并重复最后一帧
            selected_files = frame_files
            while len(selected_files) < self.frame_count:
                selected_files.append(frame_files[-1])
                
        # 读取帧
        frames = []
        for file_path in selected_files:
            img = cv2.imread(file_path)
            if img is not None:
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