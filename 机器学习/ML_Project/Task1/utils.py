"""
工具函数模块
"""
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def evaluate(model, dataloader):
    """评估模型性能"""
    model.training = False
    all_preds = []
    all_labels = []
    
    # 获取模型所在的设备
    # 假设所有参数都在同一个设备上
    device = next(iter(model.params.values())).device
    
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            scores = model.forward(imgs)
            preds = torch.argmax(scores, dim=1)
            
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


def save_model(model, path):
    """保存模型参数"""
    torch.save(model.params, path)


def load_model(model, path):
    """加载模型参数"""
    # 获取当前模型所在的设备
    device = next(iter(model.params.values())).device
    params = torch.load(path, map_location=device)
    
    # 关键修复：不仅加载初始化的参数，还要加载动态生成的参数（如 BN 的 running_mean）
    for key in params:
        if key not in model.params:
            # 如果模型中没有这个键（通常是 BN 的运行统计量），则创建它
            model.params[key] = params[key].clone().to(device)
        else:
            # 如果已有，则拷贝数值
            model.params[key].copy_(params[key])
    return model
