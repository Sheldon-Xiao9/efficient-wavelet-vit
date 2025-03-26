import argparse
import os
import time
import yaml # type: ignore
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (  # type: ignore
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)
from tqdm import tqdm

from config.data_loader import FaceForensicsLoader, CelebDFLoader
from config.transforms import get_transforms
from config.focal_loss import BinaryFocalLoss
from network.model import DeepfakeDetector
from utils.visualization import EvalVisualization

# 参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector")
    parser.add_argument("--root", "--r", type=str, default="/path/to/dataset", 
                        help="Dataset root directory")
    parser.add_argument("--model-path", "--mp", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", "--o", type=str, default="./output/eval", 
                        help="Output directory for results")
    parser.add_argument("--batch-size", "--bs", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--dim", "--d", type=int, default=128,
                        help="Feature dimension")
    parser.add_argument("--frame-count", "--fc", type=int, default=30,
                        help="Number of frames per video")
    parser.add_argument("--dataset", "--ds", type=str, default="ff++",
                        choices=["ff++", "celeb-df"],
                        help="Dataset to evaluate")
    parser.add_argument("--visualize", "--v", action="store_true",
                        help="Generate evaluation visualizations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def load_model(model_path, dim=128, device="cuda"):
    """加载预训练模型"""
    print(f"Loading model from {model_path}...")
    model = DeepfakeDetector(in_channels=3, dim=dim, dama_dim=dim).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError(f"Could not load model from {model_path}")
        
    model.eval()
    return model


def get_dataloader(args):
    """获取数据加载器"""
    print(f"Loading {args.dataset} dataset...")
    transforms = get_transforms()
    
    if args.dataset == "ff++":
        dataset = FaceForensicsLoader(
            root=args.root,
            split="test",
            frame_count=args.frame_count,
            transform=transforms['test']
        )
    elif args.dataset == "celeb-df":
        dataset = CelebDFLoader(
            root=args.root,
            split="test",
            frame_count=args.frame_count,
            transform=transforms['test']
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
def evaluate(model, dataloader, device="cuda", args=None):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = BinaryFocalLoss(alpha=0.2, gamma=2)
    
    print("Evaluating model on the test set...")
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames, batch_size=args.batch_size, ablation='dynamic')
            
            loss, _ = criterion(outputs['logits'], labels)
            test_loss += loss.item() * frames.size(0)
            
            # 收集预测结果
            probs = torch.softmax(outputs['logits'], dim=-1)
            all_preds.extend(probs[:, 1].detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    
    test_loss /= len(dataloader.dataset)
    
    binary_preds = [1 if p >= 0.55 else 0 for p in all_preds]
    
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy_score(all_labels, binary_preds),
        'auc': roc_auc_score(all_labels, all_preds),
        'precision': precision_score(all_labels, binary_preds),
        'recall': recall_score(all_labels, binary_preds),
        'f1': f1_score(all_labels, binary_preds),
        'ap': average_precision_score(all_labels, all_preds),
        'conf_matrix': confusion_matrix(all_labels, binary_preds)
    }
    
    return metrics, np.array(all_preds), np.array(all_labels)

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*50)
    
    model = load_model(args.model_path, dim=args.dim, device=device)
    dataloader = get_dataloader(args)
    
    # 评估模型
    start_time = time.time()
    metrics, preds, labels = evaluate(model, dataloader, device=device, args=args)
    eval_time = time.time() - start_time
    
    # 输出评估结果
    print("\n"+"="*50)
    print(f"Evaluation complete in {eval_time:.2f}s")
    print("Results:")
    print(f"Test Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"AUC:            {metrics['auc']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"Average Precision: {metrics['ap']:.4f}")
    print(f"Confusion Matrix:")
    print(metrics['conf_matrix'])
    print("="*50)
    
    # 保存评估结果到yaml文件
    output_path = os.path.join(args.output, "eval_results.yaml")
    with open(output_path, "w") as f:
        f.write(f"# Evaluation results for {args.model_path}\n")
        serialized_metrics = {
            k: float(v) if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        yaml.dump(serialized_metrics, f, default_flow_style=False)
        
    print(f"Saved evaluation results to {output_path}")
    
    if args.visualize:
        print("Generating evaluation visualizations...")
        viz = EvalVisualization(args.output)
        viz.plot_metrics(metrics, labels, preds)
        print(f"Saved visualizations to {args.output}")

if __name__ == "__main__":
    main()