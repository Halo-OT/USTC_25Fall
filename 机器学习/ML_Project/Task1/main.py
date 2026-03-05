"""
训练脚本 - 支持 GPU, CNN/ResNet 架构, 以及详细日志记录
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import time

from dataset import create_dataloader
from model import SimpleCNN, ResNet, cross_entropy_loss, focal_loss
from optimizer import Adam
from utils import evaluate, save_model, load_model


def train(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据路径
    train_img_dir = os.path.join(args.data_path, 'train', 'img')
    train_txt_dir = os.path.join(args.data_path, 'train', 'txt')
    
    # 初始化日志文件
    log_file = open('train.log', 'a')
    log_file.write(f"\n{'='*20} 开始新训练 {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n")
    log_file.write(f"参数: {args}\n")
    log_file.flush()
    
    # 初始化模型
    print(f"初始化模型: {args.model_type}...")
    if args.model_type == 'cnn':
        model = SimpleCNN()
    elif args.model_type == 'resnet':
        model = ResNet()
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    model.to(device)
    
    # 加载 Checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"正在从 {args.resume} 恢复训练...")
            load_model(model, args.resume)
            start_epoch = args.start_epoch
            print(f"将从 Epoch {start_epoch + 1} 开始训练")
        else:
            print(f"警告: 未找到 checkpoint 文件 {args.resume}，将从头开始训练。")
    
    # 优化器
    optimizer = Adam(model.get_params(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 类别权重 (处理不平衡)
    # 正常: 37841, 缺陷: 3870 -> 权重约 1:10
    class_weights = torch.Tensor([1.0, 4.0]).to(device)
    
    best_f1 = 0.0
    
    print(f"开始训练, epochs: {args.epochs}, batch_size: {args.batch_size}")
    
    for epoch in range(start_epoch, args.epochs):
        model.training = True
        epoch_loss = 0.0
        
        # 学习率衰减 (手动实现)
        if epoch > 0 and epoch % 5 == 0:
            optimizer.lr *= 0.5
            print(f"学习率衰减至: {optimizer.lr}")
        
        # 创建数据加载器
        dataloader = create_dataloader(
            train_img_dir, train_txt_dir, 
            batch_size=args.batch_size, 
            shuffle=True,
            limit=args.limit,
            is_train=True,
            img_size=args.img_size
        )
        
        # 统计总步数用于进度条
        total_steps = len(dataloader)
        pbar = tqdm(dataloader, total=total_steps, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (imgs, labels, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 前向传播
            scores = model.forward(imgs)
            
            # 计算损失和梯度
            if args.loss_type == 'focal':
                loss, dout = focal_loss(scores, labels, class_weights=class_weights)
            else:
                loss, dout = cross_entropy_loss(scores, labels, class_weights=class_weights)
            
            # 反向传播
            model.backward(dout)
            
            # 更新参数
            optimizer.step(model.get_grads())
            
            # 清零梯度
            optimizer.zero_grad(model.get_grads())
            
            epoch_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # 每 100 步记录一次日志
            if step % 100 == 0:
                log_msg = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}\n"
                log_file.write(log_msg)
                log_file.flush()
        
        avg_loss = epoch_loss / (step + 1)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        log_file.write(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}\n")
        
        # 评估
        eval_limit = 1000
        eval_dataloader = create_dataloader(
            train_img_dir, train_txt_dir, 
            batch_size=args.batch_size, 
            shuffle=False,
            limit=eval_limit,
            is_train=False,
            img_size=args.img_size
        )
        metrics = evaluate(model, eval_dataloader)
        eval_msg = f"评估结果: Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}\n"
        print(eval_msg)
        log_file.write(eval_msg)
        log_file.flush()
        
        # 保存定期 checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
        save_model(model, checkpoint_path)
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_model(model, args.save_path)
            print(f"保存最佳模型 (F1: {best_f1:.4f})")
            log_file.write(f"保存最佳模型 (F1: {best_f1:.4f})\n")
            log_file.flush()

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值改为 'data'，适应在根目录下运行
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['cnn', 'resnet'])
    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint 文件路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='恢复训练的起始 epoch (0-indexed)')
    parser.add_argument('--limit', type=int, default=None, help='限制训练集大小用于快速测试')
    parser.add_argument('--img_size', type=int, default=224, help='图像缩放尺寸')
    
    args = parser.parse_args()
    train(args)
