"""
优化器模块 - 手动实现参数更新
"""
import torch


class Adam:
    """Adam优化器"""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = {}
        self.v = {}
        self.t = 0
        
        for key in params:
            self.m[key] = torch.zeros_like(params[key])
            self.v[key] = torch.zeros_like(params[key])
    
    def step(self, grads):
        self.t += 1
        
        for key in self.params:
            if key in grads and grads[key] is not None:
                grad = grads[key]
                
                # 权重衰减 (L2 正则化)
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * self.params[key]
                
                # 更新一阶矩和二阶矩
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                
                # 偏差修正 (Bias Correction)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # 参数更新
                self.params[key] = self.params[key] - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self, grads):
        for key in grads:
            grads[key].zero_()


class SGD:
    """SGD优化器"""
    
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.velocity = {}
        for key in params:
            self.velocity[key] = torch.zeros_like(params[key])
    
    def step(self, grads):
        for key in self.params:
            if key in grads:
                grad = grads[key]
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * self.params[key]
                
                if self.momentum != 0:
                    self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grad
                    self.params[key] = self.params[key] + self.velocity[key]
                else:
                    self.params[key] = self.params[key] - self.lr * grad
    
    def zero_grad(self, grads):
        for key in grads:
            grads[key].zero_()
