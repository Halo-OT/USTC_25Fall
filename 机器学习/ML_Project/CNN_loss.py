"""
损失函数和优化器
"""
import numpy as np
import math

# 尝试导入torch用于GPU加速
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except:
    USE_GPU = False


def binary_cross_entropy_loss(pred, target, pos_weight=1.0):
    """
    二元交叉熵损失（手动实现，已废弃，建议使用binary_cross_entropy_with_logits_loss）
    pred: (batch_size, 1) 预测概率
    target: (batch_size, 1) 真实标签 (0或1)
    pos_weight: 正样本权重（用于处理类别不平衡）
    """
    # 防止数值不稳定
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    
    # 计算损失
    loss = -(pos_weight * target * np.log(pred) + (1 - target) * np.log(1 - pred))
    loss = np.mean(loss)
    
    # 计算梯度
    grad = -(pos_weight * target / pred - (1 - target) / (1 - pred)) / pred.shape[0]
    
    return loss, grad


def binary_cross_entropy_with_logits_loss(logits, target, pos_weight=1.0, neg_weight=1.0):
    """
    BCEWithLogitsLoss（数值稳定版本，CPU实现）
    支持非对称权重：neg_weight 用于惩罚 FP（假阳性），pos_weight 用于处理类别不平衡
    
    Args:
        logits: 模型输出的logits (batch_size, 1)，未经过sigmoid
        target: 真实标签 (batch_size, 1)
        pos_weight: 正样本权重，用于处理类别不平衡
        neg_weight: 负样本权重，用于惩罚 FP（假阳性），推荐值 1.5~2.0
    
    Returns:
        loss: 标量损失值
        grad: 对logits的梯度 (batch_size, 1)
    """
    # 数值稳定的sigmoid计算
    logits_clamped = np.clip(logits, -500, 500)
    sigmoid_logits = 1 / (1 + np.exp(-logits_clamped))
    
    # 数值稳定的损失计算
    # loss = neg_weight * (1 - y) * BCE_neg + pos_weight * y * BCE_pos
    max_logits = np.maximum(logits, 0)
    neg_abs_logits = -np.abs(logits)
    log_exp_term = np.log1p(np.exp(neg_abs_logits))  # log(1 + exp(-abs(x)))
    
    # 非对称加权BCE损失公式
    # loss = neg_weight * (1-y) * [max(x,0) + log(1+exp(-|x|))] + pos_weight * y * [max(x,0) - x + log(1+exp(-|x|))]
    loss = neg_weight * (1 - target) * (max_logits + log_exp_term) + pos_weight * target * (max_logits - logits + log_exp_term)
    loss = np.mean(loss)
    
    # 计算梯度 dL/dlogits
    # 对于 y=0: dL/dx = neg_weight * sigmoid(x)
    # 对于 y=1: dL/dx = pos_weight * (sigmoid(x) - 1)
    # 统一形式: dL/dx = sigmoid(x) * [neg_weight * (1-y) + pos_weight * y] - pos_weight * y
    grad = sigmoid_logits * (neg_weight * (1 - target) + pos_weight * target) - pos_weight * target
    grad = grad / target.shape[0]  # 平均化
    
    return loss, grad


class AdamOptimizer:
    """Adam优化器（手动实现，支持GPU）"""
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # 为每个参数创建momentum和variance
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        
        # 检测第一个参数是否为torch tensor
        self.use_torch = False
        if params:
            first_param = list(params.values())[0]
            if USE_GPU and isinstance(first_param, torch.Tensor):
                self.use_torch = True
        
        # 初始化
        for name, param in params.items():
            if self.use_torch:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
            else:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
    
    def step(self):
        """在每个batch开始时调用，更新t"""
        self.t += 1
    
    def get_update(self, name, param, grad):
        """获取参数更新值（支持numpy和torch tensor）"""
        if grad is None:
            if self.use_torch:
                return torch.zeros_like(param)
            else:
                return np.zeros_like(param)
        
        # 梯度裁剪（防止单个参数的梯度过大）
        max_grad_norm = 1.0
        if self.use_torch:
            grad_norm = torch.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
        else:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
            
        # 更新一阶矩估计
        if name not in self.m:
            if self.use_torch:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
            else:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
        
        if self.use_torch:
            # 使用torch操作
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 计算更新
            update = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        else:
            # 使用numpy操作
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # 计算更新
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return update


class SGDOptimizer:
    """SGD优化器（带momentum）"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}
    
    def get_update(self, name, grad):
        """获取参数更新值"""
        if name not in self.velocity:
            self.velocity[name] = np.zeros_like(grad)
        
        self.velocity[name] = self.momentum * self.velocity[name] + self.lr * grad
        return self.velocity[name]

