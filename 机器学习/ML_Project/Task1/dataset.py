"""
数据集加载模块
负责加载图像和标签，进行预处理
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from augmentation import get_train_augmentation


class GlassDefectDataset:
    """玻璃缺陷检测数据集"""
    
    def __init__(self, img_dir, txt_dir=None, transform=None, limit=None, img_size=224):
        """
        Args:
            img_dir: 图像文件夹路径
            txt_dir: 标签文件夹路径（测试时为None）
            transform: 数据增强函数
            limit: 限制数据集大小（用于快速实验）
            img_size: 图像缩放尺寸
        """
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.img_size = img_size
        
        # 获取所有图像文件名
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
            
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # 限制数据集大小
        if limit is not None and limit < len(self.img_files):
            self.img_files = self.img_files[:limit]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        Returns:
            img_tensor: 图像张量 shape (3, 320, 320)
            label: 二分类标签 (0: 无缺陷, 1: 有缺陷)
            img_name: 图像文件名（不含扩展名）
        """
        img_name = self.img_files[idx]
        img_name_without_ext = os.path.splitext(img_name)[0]
        
        # 加载图像
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # 调整大小 (降低分辨率以加速训练)
        img = img.resize((self.img_size, self.img_size))
        
        # 图像转为numpy数组并归一化
        img_array = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        
        # 转换为 (C, H, W) 格式
        img_array = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # 转换为torch张量
        img_tensor = torch.from_numpy(img_array)
        
        # 数据增强
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        # 判断是否有缺陷（检查txt文件是否存在）
        if self.txt_dir is not None:
            txt_path = os.path.join(self.txt_dir, img_name_without_ext + '.txt')
            label = 1 if os.path.exists(txt_path) else 0
        else:
            label = -1  # 测试时无标签
        
        return img_tensor, label, img_name_without_ext


def create_dataloader(img_dir, txt_dir=None, batch_size=32, shuffle=True, limit=None, is_train=True, img_size=224):
    """创建数据加载器"""
    transform = get_train_augmentation() if is_train else None
    dataset = GlassDefectDataset(img_dir, txt_dir, transform=transform, limit=limit, img_size=img_size)
    
    # 使用 PyTorch 的 DataLoader 实现多进程加载，大幅提升速度
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,  # 多进程加载
        pin_memory=True # 加速数据传输到 GPU
    )
    return dataloader
