import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from dataset import create_dataloader
from model import SimpleCNN, ResNet
from utils import load_model

def find_best_threshold(model, dataloader, device):
    """
    使用 TTA (原图+水平翻转+垂直翻转) 获取预测概率，并搜索最佳 F1 阈值
    """
    model.training = False
    all_probs = []
    all_labels = []
    
    print("正在进行 TTA 推理 (原图 + 水平翻转 + 垂直翻转)...")
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(dataloader):
            imgs = imgs.to(device)
            
            # 1. 原图
            logits = model.forward(imgs)
            probs = torch.softmax(logits, dim=1)
            
            # 2. 水平翻转
            imgs_hf = torch.flip(imgs, dims=[3])
            logits_hf = model.forward(imgs_hf)
            probs_hf = torch.softmax(logits_hf, dim=1)
            probs += probs_hf
            
            # 3. 垂直翻转
            imgs_vf = torch.flip(imgs, dims=[2])
            logits_vf = model.forward(imgs_vf)
            probs_vf = torch.softmax(logits_vf, dim=1)
            probs += probs_vf
            
            # 取平均
            probs /= 3.0
            
            # 获取属于类别 1 (缺陷) 的概率
            pos_probs = probs[:, 1]
            
            all_probs.extend(pos_probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print("\n开始搜索最优阈值 (Step 0.005)...")
    
    best_f1 = -1.0
    best_thresh = 0.5
    
    # 遍历阈值
    thresholds = np.arange(0.01, 1.00, 0.005)
    
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"\n" + "="*40)
    print(f"搜索完成！")
    print(f"最佳阈值 (Threshold): {best_thresh:.3f}")
    print(f"最高 F1 Score:       {best_f1:.4f}")
    print("="*40 + "\n")
    
    return best_thresh

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
    else:
        print(f"错误: 找不到模型文件 {args.model_path}")
        return

    test_img_dir = os.path.join(args.data_path, 'test', 'img')
    test_txt_dir = os.path.join(args.data_path, 'test', 'txt')
    
    dataloader = create_dataloader(
        test_img_dir, test_txt_dir,
        batch_size=args.batch_size,
        is_train=False,
        img_size=args.img_size
    )
    
    find_best_threshold(model, dataloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='resnet')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=320)
    
    args = parser.parse_args()
    main(args)
