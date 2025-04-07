"""
Extracts images from Celeb-DF-v2 videos
改自ff++的官方提取器
"""
import os
import numpy as np
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


DATASET_PATHS = {
    'real': 'Celeb-real',
    'fake': 'Celeb-synthesis'
}


def extract_frames(data_path, output_path, method='cv2', n_frames=300):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < n_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()
        for idx, frame_idx in enumerate(indices):
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(idx)), image)
            del image
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))

def read_testing_videos(testing_file):
    """
    从测试视频列表文件中读取视频ID
    
    :param testing_file: 测试视频列表文件路径
    :return: 真实与伪造视频ID字典
    """
    test_videos = {'real': [], 'fake': []}
    skipped_youtube = 0
    
    with open(testing_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):  # 跳过空行和注释
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
            
            # 在Celeb-DF中: 1=真实, 0=伪造
            if label == '1' and 'real' in video_path.lower():
                test_videos['real'].append(video_id)
            elif label == '0' and 'synthesis' in video_path.lower():
                test_videos['fake'].append(video_id)
    
    print(f"Skipped {skipped_youtube} YouTube videos")
    return test_videos

def extract_celebdf_testing_videos(data_path, testing_file, output_path):
    """
    仅提取测试视频列表中的视频帧
    
    :param data_path: 数据集根目录
    :param testing_file: 测试视频列表文件
    :param output_path: 输出目录
    :param compression: 压缩级别目录名
    """
    # 读取测试视频列表
    test_videos = read_testing_videos(testing_file)
    print(f"Found {len(test_videos['real'])} real and {len(test_videos['fake'])} fake test videos")
    
    # 处理真实视频
    for cat, video_ids in test_videos.items():
        videos_path = join(data_path, DATASET_PATHS[cat])
        images_path = join(output_path, 'celebdf', 'frames', DATASET_PATHS[cat])
        
        print(f"Processing {len(video_ids)} {cat} test videos")
        os.makedirs(images_path, exist_ok=True)
        
        for video_id in tqdm(video_ids, desc=f"Extracting {cat} videos"):
            video_path = join(videos_path, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                print(f"Warning: Video {video_path} not found")
                continue
                
            output_folder = join(images_path, video_id)
            extract_frames(video_path, output_folder, n_frames=300)  # 每个视频提取300帧

def extract_celebdf_videos(data_path, category, output_path):
    """
    提取Celeb-DF视频帧
    
    :param data_path: 数据集根目录
    :param category: 'real' 或 'fake' 或 'all'
    :param output_path: 输出目录
    :param compression: 压缩级别目录名(可选)
    """
    categories_to_extract = [category] if category != 'all' else ['real', 'fake']
    
    for cat in categories_to_extract:
        videos_path = join(data_path, DATASET_PATHS[cat])
        images_path = join(output_path, 'celebdf', 'frames', DATASET_PATHS[cat])
        
        print(f"Processing {cat} videos from {videos_path} to {images_path}")
        
        # 确保输出目录存在
        os.makedirs(images_path, exist_ok=True)
        
        # 遍历目录下所有视频
        videos = [v for v in os.listdir(videos_path) if v.endswith('.mp4')]
        for video in tqdm(videos, desc=f"Extracting {cat} videos"):
            video_path = join(videos_path, video)
            video_id = video.split('.')[0]  # 移除.mp4后缀
            output_folder = join(images_path, video_id)
            extract_frames(video_path, output_folder, n_frames=300)  # 每个视频提取300帧


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--category', type=str, choices=['real', 'fake', 'all'], default='all')
    p.add_argument('--output_path', type=str, default='data/frames/celeb-df')
    p.add_argument('--testing_file', type=str, default='Celeb-DF-v2/List_of_testing_videos.txt')
    args = p.parse_args()

    # 提取Celeb-DF视频帧
    if args.testing_file:
        extract_celebdf_testing_videos(args.data_path, args.testing_file, args.output_path)
    else:
        # 否则使用原始函数提取所有视频
        extract_celebdf_videos(args.data_path, args.category, args.output_path)
    print('Done!')
