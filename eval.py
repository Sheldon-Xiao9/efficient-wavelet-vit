import argparse
import os
import time
import json
import numpy as np
import pandas as pd # type: ignore
import torch
from torch.nn import BCEWithLogitsLoss
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

from config.data_loader import FaceForensicsLoader, CelebDFLoader, DiffusionLoader
from config.transforms import get_transforms
from config.focal_loss import BinaryFocalLoss
from train import combined_loss
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
    # parser.add_argument("--dim", "--d", type=int, default=128,
    #                     help="Feature dimension")
    parser.add_argument("--frame-count", "--fc", type=int, default=300,
                        help="Number of frames per video")
    parser.add_argument("--dataset", "--ds", type=str, default="ff++",
                        choices=["ff++", "celeb-df", "diffusion"],
                        help="Dataset to evaluate")
    parser.add_argument("--test-list", "--tl", type=str, 
                        default="Celeb-DF-v2/List_of_testing_videos.txt",
                        help="Path to testing video list for Celeb-DF")
    parser.add_argument("--ablation", "-a", type=str, 
                        default="dynamic",
                        choices=["dynamic", "sfe_only", "sfe_mwt"],
                        help="Ablation study type")
    parser.add_argument("--visualize", "--v", action="store_true",
                        help="Generate evaluation visualizations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def load_model(model_path, dim=128, device="cuda"):
    """加载预训练模型"""
    print(f"Loading model from {model_path}...")
    model = DeepfakeDetector(in_channels=3, dama_dim=dim).to(device)
    
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
            transform=transforms['test'],
            single_method=getattr(args, 'single_method', None)
        )
    elif args.dataset == "celeb-df":
        dataset = CelebDFLoader(
            root=args.root,
            split=["test"],
            frame_count=args.frame_count,
            transform=transforms['test'],
            testing_file=args.test_list
        )
    elif args.dataset == "diffusion":
        dataset = DiffusionLoader(
            root=args.root,
            frame_count=args.frame_count,
            transform=transforms['test'],
            single_method=getattr(args, 'single_method', None)
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
    test_orth_loss = []
    criterion = BCEWithLogitsLoss()
    
    print("Evaluating model on the test set...")
    
    ablation_mode = args.ablation if hasattr(args, 'ablation') else 'dynamic'
    print(f"Using ablation mode: {ablation_mode}")
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Testing"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames, batch_size=args.batch_size, ablation=ablation_mode)
            
            if ablation_mode == 'dynamic':
                loss, losses = combined_loss(outputs, labels, criterion, epoch=1, max_epochs=1)
                test_orth_loss.append(losses['orth_loss'])
            else:
                # 对于其他模式，使用简单的BCEWithLogitsLoss
                loss = criterion(outputs['logits'].squeeze(), labels.float())
            test_loss += loss.item() * frames.size(0)
            
            # 收集预测结果
            probs = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            if ablation_mode == 'dynamic':
                test_orth_loss.append(losses['orth_loss'])
            
    
    test_loss /= len(dataloader.dataset)
    
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    
    metrics = {
        'loss': test_loss,
        'orth_loss': test_orth_loss,
        'accuracy': accuracy_score(all_labels, binary_preds),
        'auc': roc_auc_score(all_labels, all_preds),
        'precision': precision_score(all_labels, binary_preds),
        'recall': recall_score(all_labels, binary_preds),
        'f1': f1_score(all_labels, binary_preds),
        'ap': average_precision_score(all_labels, all_preds),
        'conf_matrix': confusion_matrix(all_labels, binary_preds),
        'labels': all_labels,
        'preds': all_preds
    }
    
    return metrics, np.array(all_preds), np.array(all_labels)

def print_metrics(metrics):
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
    
    # 按方法评估
    if args.dataset == "ff++":
        methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter']
        all_results = {}
        
        # 先添加一个整体评估
        print("\n" + "="*50)
        print("Evaluating on all methods combined")
        dataloader = get_dataloader(args)
        start_time = time.time()
        metrics, preds, labels = evaluate(model, dataloader, device=device, args=args)
        eval_time = time.time() - start_time
        print(f"Evaluation on all methods complete in {eval_time:.2f}s")
        all_results['All'] = metrics
        
        # 输出整体评估结果
        print_metrics(metrics)
        
        # 按照每种方法单独评估
        for method in methods:
            print("\n" + "="*50)
            print(f"Evaluating on {method}")
            
            # 为特定方法创建数据加载器
            args.single_method = method  # 添加这个属性
            method_dataloader = get_dataloader(args)
            
            # 评估特定方法
            start_time = time.time()
            method_metrics, method_preds, method_labels = evaluate(model, method_dataloader, device=device, args=args)
            eval_time = time.time() - start_time
            print(f"Evaluation on {method} complete in {eval_time:.2f}s")
            all_results[method] = method_metrics
            
            # 输出特定方法的评估结果
            print_metrics(method_metrics)
        
        # 将结果保存为CSV文件
        results_df = []
        for method_name, metrics in all_results.items():
            # 提取主要指标
            row = {
                'Method': method_name,
                'Loss': metrics['loss'],
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'AP': metrics['ap'],
                'TN': metrics['conf_matrix'][0, 0],
                'FP': metrics['conf_matrix'][0, 1],
                'FN': metrics['conf_matrix'][1, 0],
                'TP': metrics['conf_matrix'][1, 1]
            }
            results_df.append(row)
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(results_df)
        output_path = os.path.join(args.output, "eval_results.csv")
        df.to_csv(output_path, index=False)
        
        # 另存为混淆矩阵详细信息
        conf_matrices = {}
        for method_name, metrics in all_results.items():
            conf_matrices[f"{method_name}_matrix"] = metrics['conf_matrix'].tolist()
        
        # 保存为JSON格式
        json_path = os.path.join(args.output, "confusion_matrices.json")
        with open(json_path, 'w') as f:
            json.dump(conf_matrices, f, indent=2)
            
        print(f"Saved evaluation results to {output_path}")
        
        if args.visualize:
            print("Generating evaluation visualizations...")
            
            # 为整体结果创建可视化
            all_viz_dir = os.path.join(args.output, "visualizations", "all_methods") 
            os.makedirs(all_viz_dir, exist_ok=True)
            all_viz = EvalVisualization(all_viz_dir)
            all_viz.plot_metrics(all_results['All'], labels, preds, metrics['orth_loss'])
            
            # 为每种方法创建单独的可视化
            for method in methods:
                method_viz_dir = os.path.join(args.output, "visualizations", method)
                os.makedirs(method_viz_dir, exist_ok=True)
                method_viz = EvalVisualization(method_viz_dir)
                
                # 获取该方法的结果
                method_metrics = all_results[method]
                method_labels = np.array(method_metrics.get('labels', []))
                method_preds = np.array(method_metrics.get('preds', []))
                method_viz.plot_metrics(method_metrics, method_labels, method_preds, method_metrics['orth_loss'])
                
            print(f"Saved visualizations to {os.path.join(args.output, 'visualizations')}")
    
    # 如果是CelebDF数据集
    elif args.dataset == "celeb-df":
        dataloader = get_dataloader(args)
    
        # 评估模型
        start_time = time.time()
        metrics, preds, labels = evaluate(model, dataloader, device=device, args=args)
        eval_time = time.time() - start_time
        
        # 输出评估结果
        print("\n"+"="*50)
        print(f"Evaluation complete in {eval_time:.2f}s")
        
        print_metrics(metrics)
        
        # 保存为CSV
        results = {
            'Loss': metrics['loss'],
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AP': metrics['ap'],
            'TN': metrics['conf_matrix'][0, 0],
            'FP': metrics['conf_matrix'][0, 1],
            'FN': metrics['conf_matrix'][1, 0],
            'TP': metrics['conf_matrix'][1, 1]
        }
        
        df = pd.DataFrame([results])
        output_path = os.path.join(args.output, "eval_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved evaluation results to {output_path}")
            
        print(f"Saved evaluation results to {output_path}")
        
        if args.visualize:
            print("Generating evaluation visualizations...")
            viz = EvalVisualization(args.output)
            orth_loss = metrics['orth_loss']
            viz.plot_metrics(metrics, labels, preds, orth_loss)
            print(f"Saved visualizations to {args.output}")
    elif args.dataset == "diffusion":
        methods = ['DDPM', 'DDIM', 'LDM']
        all_results = {}
        
        for method in methods:
            print("\n" + "="*50)
            print(f"Evaluating on {method}")
            
            # 为特定方法创建数据加载器
            args.single_method = method
            method_dataloader = get_dataloader(args)
            
            # 评估特定方法
            start_time = time.time()
            method_metrics, method_preds, method_labels = evaluate(model, method_dataloader, device=device, args=args)

if __name__ == "__main__":
    main()