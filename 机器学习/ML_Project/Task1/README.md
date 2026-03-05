# Glass Defect Detection - Task 1

本项目是基于 PyTorch 实现的玻璃缺陷检测系统（Task 1）。核心特点是**手动实现了所有模型层的反向传播梯度计算**，不依赖 `torch.autograd`，并针对高性能 GPU（如 V100/RTX 3090）进行了向量化优化。

## 核心功能

1.  **模型架构**:
    *   `SimpleCNN`: 包含卷积、批归一化 (BN)、ReLU、最大池化和全局平均池化 (GAP)。
    *   `ResNet`: 包含残差块 (Residual Blocks) 和跳跃连接，支持更深的网络。
    *   **手动反向传播**: 所有层（Conv, BN, ReLU, Pool, GAP, FC）均手动实现梯度计算。
2.  **损失函数**:
    *   `Focal Loss`: 专门针对 1:10 的类别不平衡问题设计。
    *   `Weighted Cross Entropy`: 支持类别加权的交叉熵。
3.  **优化器**:
    *   手动实现 `Adam` 优化器，包含偏差修正 (Bias Correction) 和权重衰减 (Weight Decay)。
4.  **性能优化**:
    *   **向量化梯度**: 利用 `F.conv2d` 组卷积特性实现完全向量化的权重梯度计算。
    *   **显存管理**: 训练过程中及时清理缓存，支持大 Batch Size 训练。
    *   **GPU 加速**: 全面适配 CUDA。

## 文件结构

*   `model.py`: 模型定义与手动反向传播逻辑。
*   `dataset.py`: 数据加载与预处理（320x320 缩放）。
*   `optimizer.py`: 手动实现的 Adam 和 SGD 优化器。
*   `main.py`: 训练主脚本，支持命令行参数配置。
*   `utils.py`: 评估指标计算与模型保存/加载。
*   `augmentation.py`: 数据增强策略。

## 使用方式

### 1. 环境准备
确保安装了 `torch`, `numpy`, `Pillow`, `tqdm`, `scikit-learn`。

### 2. 训练模型
建议在有 GPU 的环境下运行。

**训练 ResNet (推荐)**:
```bash
python Task1/main.py --model_type resnet --loss_type focal --batch_size 64 --epochs 20 --lr 0.001
```

**训练 SimpleCNN**:
```bash
python Task1/main.py --model_type cnn --loss_type ce --batch_size 128 --epochs 15
```

**恢复训练**:
如果你想从某个 checkpoint 恢复训练（例如从第 2 个 epoch 之后继续）：
```bash
python Task1/main.py --model_type resnet --resume checkpoint_epoch_2.pth --start_epoch 2
```

### 3. 参数说明
*   `--model_type`: 模型选择 (`cnn` 或 `resnet`)。
*   `--loss_type`: 损失函数 (`ce` 或 `focal`)。
*   `--batch_size`: 批大小，建议 V100 使用 64-128。
*   `--lr`: 学习率，默认 0.001。
*   `--resume`: 指定要加载的 checkpoint 文件路径。
*   `--start_epoch`: 设置开始的 epoch 索引（从 0 开始，例如从 epoch 2 之后继续则填 2）。
*   `--limit`: 限制训练样本数，用于快速调试。

## 训练日志
训练过程中的 Loss 和评估指标（Accuracy, F1, Precision, Recall）会实时记录在 `train.log` 中。
