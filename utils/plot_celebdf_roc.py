import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc # type: ignore
import torch
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from eval import load_model as load_deepfake_model, get_dataloader as get_deepfake_dataloader, evaluate as evaluate_deepfake
from utils.xception.eval import load_model as load_xception_model, get_dataloader as get_xception_dataloader, evaluate as evaluate_xception

def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC curves for multiple models on Celeb-DF test set")
    parser.add_argument('--model-paths', nargs='+', required=True, help='List of model checkpoint paths')
    parser.add_argument('--model-types', nargs='+', required=True, help='List of model types (deepfake/xception)')
    parser.add_argument('--labels', nargs='+', default=None, help='List of model names for legend')
    parser.add_argument('--output', type=str, default='./output/roc_celebdf.png', help='Path to save ROC curve image')
    parser.add_argument('--root', type=str, required=True, help='Celeb-DF dataset root directory')
    parser.add_argument('--test-list', type=str, default='Celeb-DF-v2/List_of_testing_videos.txt', help='Test list file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension (for deepfake)')
    parser.add_argument('--frame-count', type=int, default=30, help='Number of frames per video')
    parser.add_argument('--ablation', type=str, default='dynamic', choices=['dynamic', 'sfe_only', 'sfe_mwt'], help='Ablation mode (for deepfake)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--per-frame', action='store_true', help='Evaluate per frame (B=1, N=1) for smoother ROC')
    return parser.parse_args()

def deepfake_per_frame_eval(model, dataloader, device, args):
    model.eval()
    all_preds = []
    all_labels = []
    # 预统计总帧数
    total_frames = 0
    for frames, _ in dataloader:
        if frames.dim() == 5:
            total_frames += frames.shape[0] * frames.shape[1]
        elif frames.dim() == 4:
            total_frames += frames.shape[0]
    # 重新遍历，单一tqdm进度条
    with torch.no_grad():
        idx = 0
        with tqdm(total=total_frames, desc="Per-frame Inference") as pbar:
            for frames, labels in dataloader:
                if frames.dim() == 5:
                    B, N, C, H, W = frames.shape
                    for b in range(B):
                        label = labels[b].item()
                        for n in range(N):
                            single_frame = frames[b, n].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]
                            outputs = model(single_frame, batch_size=1, ablation=args.ablation)
                            prob = torch.sigmoid(outputs['logits']).cpu().numpy().flatten()[0]
                            all_preds.append(prob)
                            all_labels.append(label)
                            idx += 1
                            pbar.update(1)
                elif frames.dim() == 4:
                    B, C, H, W = frames.shape
                    for b in range(B):
                        single_frame = frames[b].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, C, H, W]
                        label = labels[b].item()
                        outputs = model(single_frame, batch_size=1, ablation=args.ablation)
                        prob = torch.sigmoid(outputs['logits']).cpu().numpy().flatten()[0]
                        all_preds.append(prob)
                        all_labels.append(label)
                        idx += 1
                        pbar.update(1)
                else:
                    raise ValueError("Unsupported frame shape for deepfake model.")
    return np.array(all_preds), np.array(all_labels)

def xception_per_frame_eval(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_frames = 0
    for frames, _ in dataloader:
        if frames.dim() == 5:
            total_frames += frames.shape[0] * frames.shape[1]
        elif frames.dim() == 4:
            total_frames += frames.shape[0]
    with torch.no_grad():
        idx = 0
        with tqdm(total=total_frames, desc="Per-frame Inference") as pbar:
            for frames, labels in dataloader:
                if frames.dim() == 5:
                    B, N, C, H, W = frames.shape
                    for b in range(B):
                        label = labels[b].item()
                        for n in range(N):
                            single_frame = frames[b, n].unsqueeze(0).to(device)  # [1, C, H, W]
                            outputs = model(single_frame)
                            prob = torch.sigmoid(outputs).cpu().numpy().flatten()[0]
                            all_preds.append(prob)
                            all_labels.append(label)
                            idx += 1
                            pbar.update(1)
                elif frames.dim() == 4:
                    B, C, H, W = frames.shape
                    for b in range(B):
                        single_frame = frames[b].unsqueeze(0).to(device)  # [1, C, H, W]
                        label = labels[b].item()
                        outputs = model(single_frame)
                        prob = torch.sigmoid(outputs).cpu().numpy().flatten()[0]
                        all_preds.append(prob)
                        all_labels.append(label)
                        idx += 1
                        pbar.update(1)
                else:
                    raise ValueError("Unsupported frame shape for xception model.")
    return np.array(all_preds), np.array(all_labels)

def main():
    args = parse_args()
    model_paths = args.model_paths
    model_types = args.model_types
    if len(model_paths) != len(model_types):
        raise ValueError("The number of --model-paths and --model-types must match.")
    labels = args.labels if args.labels and len(args.labels) == len(model_paths) else [f'Model{i+1}' for i in range(len(model_paths))]
    device = args.device

    dataloader_cache = {}
    labels_cache = None

    plt.figure(figsize=(8, 6))

    for idx, (model_path, model_type, label) in enumerate(zip(model_paths, model_types, labels)):
        if model_type.lower() == 'deepfake':
            class DummyArgs:
                pass
            data_args = DummyArgs()
            data_args.root = args.root
            data_args.dataset = 'celeb-df'
            data_args.frame_count = args.frame_count
            data_args.batch_size = args.batch_size
            data_args.test_list = args.test_list
            data_args.ablation = args.ablation

            cache_key = ('deepfake', args.root, args.test_list, data_args.frame_count, data_args.batch_size, args.ablation)
            if cache_key not in dataloader_cache:
                dataloader_cache[cache_key] = get_deepfake_dataloader(data_args)
            dataloader = dataloader_cache[cache_key]

            model = load_deepfake_model(model_path, dim=args.dim, device=device)
            if args.per_frame:
                preds, labels_arr = deepfake_per_frame_eval(model, dataloader, device, data_args)
                print(f"[{label}] Per-frame mode: total frames={len(preds)}, total labels={len(labels_arr)}")
            else:
                metrics, preds, labels_arr = evaluate_deepfake(model, dataloader, device=device, args=data_args)
        elif model_type.lower() == 'xception':
            class DummyArgs:
                pass
            data_args = DummyArgs()
            data_args.root = args.root
            data_args.dataset = 'celeb-df'
            data_args.frame_count = args.frame_count
            data_args.batch_size = args.batch_size
            data_args.test_list = args.test_list

            cache_key = ('xception', args.root, args.test_list, data_args.frame_count, data_args.batch_size)
            if cache_key not in dataloader_cache:
                dataloader_cache[cache_key] = get_xception_dataloader(data_args, img_size=299)
            dataloader = dataloader_cache[cache_key]

            model, _ = load_xception_model(model_path, device=device)
            if args.per_frame:
                preds, labels_arr = xception_per_frame_eval(model, dataloader, device)
                print(f"[{label}] Per-frame mode: total frames={len(preds)}, total labels={len(labels_arr)}")
            else:
                metrics, preds, labels_arr = evaluate_xception(model, dataloader, device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        fpr, tpr, _ = roc_curve(labels_arr, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.3f})')
        if labels_cache is None:
            labels_cache = labels_arr

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Celeb-DF Test Set')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print(f'ROC curves saved to {args.output}')

if __name__ == '__main__':
    main()
