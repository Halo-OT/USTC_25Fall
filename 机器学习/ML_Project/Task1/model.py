"""
模型定义模块 - 手动实现前向和反向传播
支持 GPU (CUDA) 加速，包含 CNN 和 ResNet 架构
优化：引入 BatchNorm, GAP, He 初始化, 向量化梯度计算, 显存管理
"""
import torch
import torch.nn.functional as F
import numpy as np

def kaiming_init(shape, mode='fan_in'):
    """He 初始化 (Kaiming Initialization)"""
    if len(shape) == 4: # Conv
        fan_in = shape[1] * shape[2] * shape[3]
    else: # FC
        fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return torch.randn(*shape) * std

class BaseModel:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}
        self.training = True
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        for key in self.params:
            self.params[key] = self.params[key].to(device)
        for key in self.grads:
            self.grads[key] = self.grads[key].to(device)
        return self

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads

    def _bn_forward(self, x, gamma, beta, name, eps=1e-5, momentum=0.9):
        """BatchNorm2d 前向传播"""
        N, C, H, W = x.shape
        running_mean_key = f'{name}_running_mean'
        running_var_key = f'{name}_running_var'
        
        # 确保运行统计量在 params 中（如果 load_model 没加载到，这里做保底）
        if running_mean_key not in self.params:
            self.params[running_mean_key] = torch.zeros(C, device=self.device)
            self.params[running_var_key] = torch.ones(C, device=self.device)

        if self.training:
            # 训练模式：计算当前 batch 的均值和方差
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            
            # 更新运行均值和方差 (使用 squeeze 确保维度匹配)
            self.params[running_mean_key] = momentum * self.params[running_mean_key] + (1 - momentum) * mean.view(-1)
            self.params[running_var_key] = momentum * self.params[running_var_key] + (1 - momentum) * var.view(-1)
        else:
            # 推理模式：使用训练时累积的运行均值和方差
            mean = self.params[running_mean_key].view(1, C, 1, 1)
            var = self.params[running_var_key].view(1, C, 1, 1)

        x_hat = (x - mean) / torch.sqrt(var + eps)
        out = gamma.view(1, C, 1, 1) * x_hat + beta.view(1, C, 1, 1)
        
        if self.training:
            self.cache[f'{name}_x_hat'] = x_hat
            self.cache[f'{name}_var'] = var
            self.cache[f'{name}_mean'] = mean
            self.cache[f'{name}_eps'] = eps
            
        return out

    def _bn_backward(self, dout, name, gamma):
        """BatchNorm2d 反向传播"""
        x_hat = self.cache[f'{name}_x_hat']
        var = self.cache[f'{name}_var']
        mean = self.cache[f'{name}_mean']
        eps = self.cache[f'{name}_eps']
        N, C, H, W = dout.shape
        m = N * H * W

        dgamma = (dout * x_hat).sum(dim=(0, 2, 3))
        dbeta = dout.sum(dim=(0, 2, 3))
        
        dx_hat = dout * gamma.view(1, C, 1, 1)
        dvar = (dx_hat * (x_hat * torch.sqrt(var + eps)) * -0.5 * (var + eps)**(-1.5)).sum(dim=(0, 2, 3), keepdim=True)
        dmean = (dx_hat * -1.0 / torch.sqrt(var + eps)).sum(dim=(0, 2, 3), keepdim=True) + dvar * (-2.0 * (x_hat * torch.sqrt(var + eps))).mean(dim=(0, 2, 3), keepdim=True)
        
        dx = dx_hat / torch.sqrt(var + eps) + dvar * 2.0 * (x_hat * torch.sqrt(var + eps)) / m + dmean / m
        
        return dx, dgamma, dbeta

    def _get_conv_grad_w(self, x, dout, w_shape, padding=0, stride=1):
        """完全向量化计算卷积权重梯度"""
        # x: (N, C_in, H, W), dout: (N, C_out, H_out, W_out)
        # 将 C_in 视为 batch，N 视为 input channels
        x_p = x.transpose(0, 1) # (C_in, N, H, W)
        # 将 C_out 视为 output channels，N 视为 input channels
        dout_p = dout.transpose(0, 1) # (C_out, N, H_out, W_out)
        
        # 在 backward 中，forward 的 stride 对应这里的 dilation
        dw = F.conv2d(x_p, dout_p, padding=padding, dilation=stride)
        
        # dw shape: (C_in, C_out, kH, kW) -> 转置回 (C_out, C_in, kH, kW)
        dw = dw.transpose(0, 1)
        
        # 裁剪到原始权重大小
        return dw[:, :, :w_shape[2], :w_shape[3]]

    def _get_conv_grad_x(self, dout, w, x_shape, padding=0, stride=1, output_padding=0):
        """计算卷积输入梯度"""
        return F.conv_transpose2d(dout, w, padding=padding, stride=stride, output_padding=output_padding)

class SimpleCNN(BaseModel):
    """
    卷积神经网络 (升级版)
    架构：Conv -> BN -> ReLU -> MaxPool -> Conv -> BN -> ReLU -> MaxPool -> GAP -> FC
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Conv1: 3 -> 16
        self.params['conv1_w'] = kaiming_init((16, 3, 3, 3))
        self.params['conv1_b'] = torch.zeros(16)
        self.params['bn1_gamma'] = torch.ones(16)
        self.params['bn1_beta'] = torch.zeros(16)
        
        # Conv2: 16 -> 32
        self.params['conv2_w'] = kaiming_init((32, 16, 3, 3))
        self.params['conv2_b'] = torch.zeros(32)
        self.params['bn2_gamma'] = torch.ones(32)
        self.params['bn2_beta'] = torch.zeros(32)
        
        # FC: GAP 之后输出维度是 32
        self.params['fc_w'] = kaiming_init((32, num_classes))
        self.params['fc_b'] = torch.zeros(num_classes)
        
        for key in self.params:
            if 'running' not in key:
                self.grads[key] = torch.zeros_like(self.params[key])

    def forward(self, x):
        self.cache['x'] = x
        
        # Layer 1
        out1 = F.conv2d(x, self.params['conv1_w'], self.params['conv1_b'], padding=1)
        self.cache['out1'] = out1
        bn1 = self._bn_forward(out1, self.params['bn1_gamma'], self.params['bn1_beta'], 'bn1')
        self.cache['bn1'] = bn1
        act1 = F.relu(bn1)
        self.cache['act1'] = act1
        pool1, pool1_idx = F.max_pool2d(act1, kernel_size=2, stride=2, return_indices=True)
        self.cache['pool1'] = pool1
        self.cache['pool1_idx'] = pool1_idx
        
        # Layer 2
        out2 = F.conv2d(pool1, self.params['conv2_w'], self.params['conv2_b'], padding=1)
        self.cache['out2'] = out2
        bn2 = self._bn_forward(out2, self.params['bn2_gamma'], self.params['bn2_beta'], 'bn2')
        self.cache['bn2'] = bn2
        act2 = F.relu(bn2)
        self.cache['act2'] = act2
        pool2, pool2_idx = F.max_pool2d(act2, kernel_size=2, stride=2, return_indices=True)
        self.cache['pool2'] = pool2
        self.cache['pool2_idx'] = pool2_idx
        
        # GAP
        gap = pool2.mean(dim=(2, 3))
        self.cache['gap'] = gap
        
        # FC
        logits = torch.matmul(gap, self.params['fc_w']) + self.params['fc_b']
        return logits

    def backward(self, dout):
        # FC
        self.grads['fc_w'] = torch.matmul(self.cache['gap'].T, dout)
        self.grads['fc_b'] = torch.sum(dout, dim=0)
        dgap = torch.matmul(dout, self.params['fc_w'].T)
        
        # GAP Backward
        N, C, H, W = self.cache['pool2'].shape
        dpool2 = dgap.view(N, C, 1, 1).expand(N, C, H, W) / (H * W)
        
        # Layer 2 Backward
        dact2 = F.max_unpool2d(dpool2, self.cache['pool2_idx'], kernel_size=2, stride=2, output_size=self.cache['act2'].shape)
        dbn2 = dact2 * (self.cache['bn2'] > 0).float()
        dout2, self.grads['bn2_gamma'], self.grads['bn2_beta'] = self._bn_backward(dbn2, 'bn2', self.params['bn2_gamma'])
        
        self.grads['conv2_w'] = self._get_conv_grad_w(self.cache['pool1'], dout2, self.params['conv2_w'].shape, padding=1)
        self.grads['conv2_b'] = torch.sum(dout2, dim=(0, 2, 3))
        dpool1 = self._get_conv_grad_x(dout2, self.params['conv2_w'], self.cache['pool1'].shape, padding=1)
        
        # Layer 1 Backward
        dact1 = F.max_unpool2d(dpool1, self.cache['pool1_idx'], kernel_size=2, stride=2, output_size=self.cache['act1'].shape)
        dbn1 = dact1 * (self.cache['bn1'] > 0).float()
        dout1, self.grads['bn1_gamma'], self.grads['bn1_beta'] = self._bn_backward(dbn1, 'bn1', self.params['bn1_gamma'])
        
        self.grads['conv1_w'] = self._get_conv_grad_w(self.cache['x'], dout1, self.params['conv1_w'].shape, padding=1)
        self.grads['conv1_b'] = torch.sum(dout1, dim=(0, 2, 3))
        
        # 显存管理：清理缓存
        self.cache.clear()

class ResNet(BaseModel):
    """
    升级版 ResNet (类似 ResNet-18 结构)
    架构：Conv0 -> Stage1(2 blocks) -> Stage2(2 blocks) -> Stage3(2 blocks) -> Stage4(2 blocks) -> GAP -> FC
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # 通道数配置
        self.channels = [16, 32, 64, 128]
        
        # Initial Conv
        self.params['conv0_w'] = kaiming_init((self.channels[0], 3, 3, 3))
        self.params['conv0_b'] = torch.zeros(self.channels[0])
        self.params['bn0_gamma'] = torch.ones(self.channels[0])
        self.params['bn0_beta'] = torch.zeros(self.channels[0])
        
        # 4 Stages, 每个 Stage 2 个 Block
        in_c = self.channels[0]
        for s, out_c in enumerate(self.channels):
            for b in range(2):
                stride = 2 if (s > 0 and b == 0) else 1
                prefix = f's{s}_b{b}_'
                
                # Conv 1
                self.params[prefix + 'c1_w'] = kaiming_init((out_c, in_c, 3, 3))
                self.params[prefix + 'c1_b'] = torch.zeros(out_c)
                self.params[prefix + 'bn1_gamma'] = torch.ones(out_c)
                self.params[prefix + 'bn1_beta'] = torch.zeros(out_c)
                
                # Conv 2
                self.params[prefix + 'c2_w'] = kaiming_init((out_c, out_c, 3, 3))
                self.params[prefix + 'c2_b'] = torch.zeros(out_c)
                self.params[prefix + 'bn2_gamma'] = torch.ones(out_c)
                self.params[prefix + 'bn2_beta'] = torch.zeros(out_c)
                
                # Skip connection (if stride > 1 or in_c != out_c)
                if stride > 1 or in_c != out_c:
                    self.params[prefix + 'skip_w'] = kaiming_init((out_c, in_c, 1, 1))
                
                in_c = out_c
        
        # FC
        self.params['fc_w'] = kaiming_init((self.channels[-1], num_classes))
        self.params['fc_b'] = torch.zeros(num_classes)
        
        for key in self.params:
            if 'running' not in key:
                self.grads[key] = torch.zeros_like(self.params[key])

    def _res_block_forward(self, x, prefix, stride=1):
        in_c = x.shape[1]
        out_c = self.params[prefix + 'c1_w'].shape[0]
        
        # Skip
        if prefix + 'skip_w' in self.params:
            skip = F.conv2d(x, self.params[prefix + 'skip_w'], stride=stride)
        else:
            skip = x
        self.cache[prefix + 'skip'] = skip
        
        # Conv 1
        c1 = F.conv2d(x, self.params[prefix + 'c1_w'], self.params[prefix + 'c1_b'], padding=1, stride=stride)
        bn1 = self._bn_forward(c1, self.params[prefix + 'bn1_gamma'], self.params[prefix + 'bn1_beta'], prefix + 'bn1')
        act1 = F.relu(bn1)
        self.cache[prefix + 'act1'] = act1
        
        # Conv 2
        c2 = F.conv2d(act1, self.params[prefix + 'c2_w'], self.params[prefix + 'c2_b'], padding=1)
        bn2 = self._bn_forward(c2, self.params[prefix + 'bn2_gamma'], self.params[prefix + 'bn2_beta'], prefix + 'bn2')
        self.cache[prefix + 'bn2'] = bn2
        
        out = F.relu(bn2 + skip)
        self.cache[prefix + 'out'] = out
        return out

    def _res_block_backward(self, dout, prefix, x_in, stride=1):
        dout_act = dout * (self.cache[prefix + 'out'] > 0).float()
        
        # Conv 2 Backward
        dbn2 = dout_act
        dc2, self.grads[prefix + 'bn2_gamma'], self.grads[prefix + 'bn2_beta'] = self._bn_backward(dbn2, prefix + 'bn2', self.params[prefix + 'bn2_gamma'])
        self.grads[prefix + 'c2_w'] = self._get_conv_grad_w(self.cache[prefix + 'act1'], dc2, self.params[prefix + 'c2_w'].shape, padding=1)
        self.grads[prefix + 'c2_b'] = torch.sum(dc2, dim=(0, 2, 3))
        dact1 = self._get_conv_grad_x(dc2, self.params[prefix + 'c2_w'], self.cache[prefix + 'act1'].shape, padding=1)
        
        # Conv 1 Backward
        dbn1 = dact1 * (self.cache[prefix + 'act1'] > 0).float()
        dc1, self.grads[prefix + 'bn1_gamma'], self.grads[prefix + 'bn1_beta'] = self._bn_backward(dbn1, prefix + 'bn1', self.params[prefix + 'bn1_gamma'])
        self.grads[prefix + 'c1_w'] = self._get_conv_grad_w(x_in, dc1, self.params[prefix + 'c1_w'].shape, padding=1, stride=stride)
        self.grads[prefix + 'c1_b'] = torch.sum(dc1, dim=(0, 2, 3))
        dx_branch1 = self._get_conv_grad_x(dc1, self.params[prefix + 'c1_w'], x_in.shape, padding=1, stride=stride, 
                                         output_padding=(1 if stride==2 else 0))
        
        # Skip Backward
        if prefix + 'skip_w' in self.params:
            dskip = dout_act
            self.grads[prefix + 'skip_w'] = self._get_conv_grad_w(x_in, dskip, self.params[prefix + 'skip_w'].shape, stride=stride)
            dx_branch2 = self._get_conv_grad_x(dskip, self.params[prefix + 'skip_w'], x_in.shape, stride=stride,
                                             output_padding=(1 if stride==2 else 0))
        else:
            dx_branch2 = dout_act
            
        return dx_branch1 + dx_branch2

    def forward(self, x):
        self.cache['x'] = x
        
        # Initial Conv
        out0 = F.conv2d(x, self.params['conv0_w'], self.params['conv0_b'], padding=1)
        bn0 = self._bn_forward(out0, self.params['bn0_gamma'], self.params['bn0_beta'], 'bn0')
        act0 = F.relu(bn0)
        self.cache['act0'] = act0
        
        # Stages
        feat = act0
        for s in range(4):
            for b in range(2):
                stride = 2 if (s > 0 and b == 0) else 1
                prefix = f's{s}_b{b}_'
                feat = self._res_block_forward(feat, prefix, stride=stride)
        
        # GAP
        gap = feat.mean(dim=(2, 3))
        self.cache['gap'] = gap
        
        # FC
        logits = torch.matmul(gap, self.params['fc_w']) + self.params['fc_b']
        return logits

    def backward(self, dout):
        # FC
        self.grads['fc_w'] = torch.matmul(self.cache['gap'].T, dout)
        self.grads['fc_b'] = torch.sum(dout, dim=0)
        dgap = torch.matmul(dout, self.params['fc_w'].T)
        
        # GAP Backward
        N, C, H, W = self.cache['s3_b1_out'].shape
        dfeat = dgap.view(N, C, 1, 1).expand(N, C, H, W) / (H * W)
        
        # Stages Backward
        for s in reversed(range(4)):
            for b in reversed(range(2)):
                stride = 2 if (s > 0 and b == 0) else 1
                prefix = f's{s}_b{b}_'
                # 获取该 block 的输入
                if b > 0:
                    x_in = self.cache[f's{s}_b{b-1}_out']
                elif s > 0:
                    x_in = self.cache[f's{s-1}_b1_out']
                else:
                    x_in = self.cache['act0']
                
                dfeat = self._res_block_backward(dfeat, prefix, x_in, stride=stride)
        
        # Initial Conv Backward
        dbn0 = dfeat * (self.cache['act0'] > 0).float()
        dout0, self.grads['bn0_gamma'], self.grads['bn0_beta'] = self._bn_backward(dbn0, 'bn0', self.params['bn0_gamma'])
        self.grads['conv0_w'] = self._get_conv_grad_w(self.cache['x'], dout0, self.params['conv0_w'].shape, padding=1)
        self.grads['conv0_b'] = torch.sum(dout0, dim=(0, 2, 3))
        
        # 显存管理
        self.cache.clear()

def focal_loss(scores, labels, alpha=0.25, gamma=2.0, class_weights=None):
    batch_size = scores.shape[0]
    probs = F.softmax(scores, dim=1)
    target_probs = probs[range(batch_size), labels.long()]
    target_probs = torch.clamp(target_probs, 1e-7, 1.0 - 1e-7)
    log_p = torch.log(target_probs)
    focal_weight = (1 - target_probs) ** gamma
    if class_weights is not None:
        alpha = class_weights[labels.long()]
    loss = -torch.mean(alpha * focal_weight * log_p)
    dout = probs.clone()
    dout[range(batch_size), labels.long()] -= 1
    if class_weights is not None:
        dout = dout * class_weights[labels.long()].unsqueeze(1)
    dout = dout * focal_weight.unsqueeze(1) / batch_size
    return loss, dout

def cross_entropy_loss(scores, labels, class_weights=None):
    batch_size = scores.shape[0]
    exp_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0])
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
    probs_clamped = torch.clamp(probs, 1e-7, 1.0)
    correct_log_probs = -torch.log(probs_clamped[range(batch_size), labels.long()])
    if class_weights is not None:
        sample_weights = class_weights[labels.long()]
        loss = torch.mean(correct_log_probs * sample_weights)
    else:
        loss = torch.mean(correct_log_probs)
    dout = probs.clone()
    dout[range(batch_size), labels.long()] -= 1
    if class_weights is not None:
        dout = dout * class_weights[labels.long()].unsqueeze(1)
    dout /= batch_size
    return loss, dout
