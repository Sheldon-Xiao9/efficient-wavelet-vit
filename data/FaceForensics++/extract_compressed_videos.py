"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
import numpy as np
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm


DATASET_PATHS = {
    'original': 'original',
    'Deepfakes': 'Deepfakes',
    'Face2Face': 'Face2Face',
    'FaceSwap': 'FaceSwap',
    'NeuralTextures': 'NeuralTextures',
    'FaceShifter': 'FaceShifter'
}
COMPRESSION = ['c0', 'c23', 'c40']


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


def extract_method_videos(data_path, dataset, output_path):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset])
    images_path = join(output_path, DATASET_PATHS[dataset])
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--output_path', type=str, default='data/frames/ff++')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))
