"""
在测试集上运行评估并输出 F1 Score
"""
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dataset import create_dataloader
from model import SimpleCNN, ResNet
from utils import load_model

def evaluate_tta(model, dataloader, device, threshold=0.5):
    """使用 TTA (测试时增强) 进行评估: 原图 + 翻转 + 旋转"""
    model.training = False
    all_preds = []
    all_labels = []
    
    print(f"正在使用强力 TTA 进行评估 (原图 + 翻转 + 旋转, 阈值={threshold})...")
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            
            # Use forward instead of __call__ just in case
            # 1. 
            scores = torch.softmax(model.forward(imgs), dim=1)
            
            # 2. H-Flip
            imgs_hf = torch.flip(imgs, dims=[3])
            scores += torch.softmax(model.forward(imgs_hf), dim=1)
            
            # 3. V-Flip
            imgs_vf = torch.flip(imgs, dims=[2])
            scores += torch.softmax(model.forward(imgs_vf), dim=1)

            # 4. R90
            imgs_r90 = torch.rot90(imgs, 1, [2, 3])
            scores += torch.softmax(model.forward(imgs_r90), dim=1)

            # 5. R180
            imgs_r180 = torch.rot90(imgs, 2, [2, 3])
            scores += torch.softmax(model.forward(imgs_r180), dim=1)

            # 6. R270
            imgs_r270 = torch.rot90(imgs, 3, [2, 3])
            scores += torch.softmax(model.forward(imgs_r270), dim=1)
            
            # Avg
            avg_scores = scores / 6.0
            
            # Thresholding
            probs = avg_scores[:, 1]
            preds = (probs > threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} | 分辨率: {args.img_size}")

    if args.model_type == 'cnn':
        model = SimpleCNN()
    else:
        model = ResNet()
    
    model.to(device)
    
    if os.path.exists(args.model_path):
        load_model(model, args.model_path)
        print(f"已加载模型: {args.model_path}")
    else:
        print(f"错误: 找不到模型文件 {args.model_path}")
        return

    test_img_dir = os.path.join(args.data_path, 'test', 'img')
    test_txt_dir = os.path.join(args.data_path, 'test', 'txt')

    dataloader = create_dataloader(
        test_img_dir, test_txt_dir,
        batch_size=args.batch_size,
        shuffle=False,
        is_train=False,
        img_size=args.img_size
    )

    if args.use_tta:
        metrics = evaluate_tta(model, dataloader, device, threshold=args.threshold)
    else:
        from utils import evaluate
        print("正在测试集上进行普通评估...")
        metrics = evaluate(model, dataloader)
    
    print("\n" + "="*40)
    print(f"测试集评估结果 ({args.model_type} | TTA: Enhance | Threshold: {args.threshold}):")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='resnet')
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    main(args)
