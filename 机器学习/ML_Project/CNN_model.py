"""
CNN模型定义
"""
import numpy as np
from layers import Conv2d, MaxPool2d, Linear, ReLU, Sigmoid, Dropout


class CNN:
    """用于二分类的CNN模型"""
    def __init__(self):
        # 输入: 340x340x3 (RGB图像)
        # 第一层: Conv + ReLU + MaxPool
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)  # 170x170
        
        # 第二层: Conv + ReLU + MaxPool
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)  # 85x85
        
        # 第三层: Conv + ReLU + MaxPool
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)  # 42x42
        
        # 第四层: Conv + ReLU + MaxPool
        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu4 = ReLU()
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)  # 21x21
        
        # 全连接层（特征数将在第一次前向传播时动态确定）
        self.fc1 = None  # 延迟初始化
        self.relu5 = ReLU()
        self.dropout = Dropout(p=0.5)
        self.fc2 = None  # 延迟初始化
        self.relu6 = ReLU()
        self.fc3 = None  # 延迟初始化
        self.sigmoid = Sigmoid()
        self._fc_initialized = False
        
    def forward(self, x, apply_sigmoid=False):
        """
        前向传播
        Args:
            x: 输入数据 (batch_size, 3, 340, 340)
            apply_sigmoid: 是否应用sigmoid激活。训练时=False（返回logits），推理时=True（返回概率）
        Returns:
            如果apply_sigmoid=False: logits (batch_size, 1)
            如果apply_sigmoid=True: 概率 (batch_size, 1)
        """
        # x: (batch_size, 3, 340, 340)
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)
        
        x = self.conv4.forward(x)
        x = self.relu4.forward(x)
        x = self.pool4.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # 动态初始化全连接层（第一次前向传播时）
        if not self._fc_initialized:
            actual_features = x_flat.shape[1]
            self.fc1 = Linear(in_features=actual_features, out_features=512)
            self.fc2 = Linear(in_features=512, out_features=128)
            self.fc3 = Linear(in_features=128, out_features=1)
            self._fc_initialized = True
            print(f"Initialized FC layers with {actual_features} input features (shape: {x.shape})")
        
        x = self.fc1.forward(x_flat)
        x = self.relu5.forward(x)
        x = self.dropout.forward(x)
        
        x = self.fc2.forward(x)
        x = self.relu6.forward(x)
        
        # FC3输出logits
        logits = self.fc3.forward(x)
        
        # 根据apply_sigmoid决定是否应用sigmoid
        if apply_sigmoid:
            return self.sigmoid.forward(logits)
        else:
            return logits
    
    def backward(self, grad_output):
        """
        反向传播
        注意：训练时forward返回logits，所以这里直接对logits求梯度，不需要经过sigmoid
        """
        # 直接对logits求梯度（BCEWithLogitsLoss的梯度已经包含了sigmoid的梯度）
        grad = self.fc3.backward(grad_output)
        grad = self.relu6.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.relu5.backward(grad)
        grad = self.fc1.backward(grad)
        
        # Reshape back - 动态获取特征图的形状
        batch_size = grad.shape[0]
        # 从fc1的输入特征数推断特征图尺寸
        fc1_input_features = self.fc1.in_features
        # fc1_input_features = 256 * h * w，所以 h * w = fc1_input_features / 256
        hw = fc1_input_features // 256
        # 假设是正方形，h = w = sqrt(hw)
        h = int(hw ** 0.5)
        w = h  # 假设是正方形
        grad = grad.reshape(batch_size, 256, h, w)
        
        grad = self.pool4.backward(grad)
        grad = self.relu4.backward(grad)
        grad = self.conv4.backward(grad)
        
        grad = self.pool3.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)
        
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        
        return grad
    
    def update(self, lr):
        """更新所有层的参数（SGD）"""
        self.conv1.update(lr)
        self.conv2.update(lr)
        self.conv3.update(lr)
        self.conv4.update(lr)
        if self._fc_initialized:
            self.fc1.update(lr)
            self.fc2.update(lr)
            self.fc3.update(lr)
    
    def update_with_adam(self, optimizer):
        """使用Adam优化器更新参数"""
        # 收集所有参数的梯度
        params = {
            'conv1_weight': self.conv1.dweight,
            'conv1_bias': self.conv1.dbias,
            'conv2_weight': self.conv2.dweight,
            'conv2_bias': self.conv2.dbias,
            'conv3_weight': self.conv3.dweight,
            'conv3_bias': self.conv3.dbias,
            'conv4_weight': self.conv4.dweight,
            'conv4_bias': self.conv4.dbias,
            'fc1_weight': self.fc1.dweight,
            'fc1_bias': self.fc1.dbias,
            'fc2_weight': self.fc2.dweight,
            'fc2_bias': self.fc2.dbias,
            'fc3_weight': self.fc3.dweight,
            'fc3_bias': self.fc3.dbias,
        }
        
        # 更新参数
        self.conv1.weight -= optimizer.get_update('conv1_weight', self.conv1.weight, self.conv1.dweight)
        self.conv1.bias -= optimizer.get_update('conv1_bias', self.conv1.bias, self.conv1.dbias)
        self.conv2.weight -= optimizer.get_update('conv2_weight', self.conv2.weight, self.conv2.dweight)
        self.conv2.bias -= optimizer.get_update('conv2_bias', self.conv2.bias, self.conv2.dbias)
        self.conv3.weight -= optimizer.get_update('conv3_weight', self.conv3.weight, self.conv3.dweight)
        self.conv3.bias -= optimizer.get_update('conv3_bias', self.conv3.bias, self.conv3.dbias)
        self.conv4.weight -= optimizer.get_update('conv4_weight', self.conv4.weight, self.conv4.dweight)
        self.conv4.bias -= optimizer.get_update('conv4_bias', self.conv4.bias, self.conv4.dbias)
        if self._fc_initialized:
            self.fc1.weight -= optimizer.get_update('fc1_weight', self.fc1.weight, self.fc1.dweight)
            self.fc1.bias -= optimizer.get_update('fc1_bias', self.fc1.bias, self.fc1.dbias)
            self.fc2.weight -= optimizer.get_update('fc2_weight', self.fc2.weight, self.fc2.dweight)
            self.fc2.bias -= optimizer.get_update('fc2_bias', self.fc2.bias, self.fc2.dbias)
            self.fc3.weight -= optimizer.get_update('fc3_weight', self.fc3.weight, self.fc3.dweight)
            self.fc3.bias -= optimizer.get_update('fc3_bias', self.fc3.bias, self.fc3.dbias)
    
    def set_training(self, training):
        """设置训练/测试模式"""
        self.dropout.training = training
    
    def get_params(self):
        """获取所有参数（用于保存）"""
        params = {
            'conv1_weight': self.conv1.weight,
            'conv1_bias': self.conv1.bias,
            'conv2_weight': self.conv2.weight,
            'conv2_bias': self.conv2.bias,
            'conv3_weight': self.conv3.weight,
            'conv3_bias': self.conv3.bias,
            'conv4_weight': self.conv4.weight,
            'conv4_bias': self.conv4.bias,
        }
        if self._fc_initialized:
            params.update({
                'fc1_weight': self.fc1.weight,
                'fc1_bias': self.fc1.bias,
                'fc2_weight': self.fc2.weight,
                'fc2_bias': self.fc2.bias,
                'fc3_weight': self.fc3.weight,
                'fc3_bias': self.fc3.bias,
            })
        return params
    
    def load_params(self, params):
        """加载参数"""
        self.conv1.weight = params['conv1_weight']
        self.conv1.bias = params['conv1_bias']
        self.conv2.weight = params['conv2_weight']
        self.conv2.bias = params['conv2_bias']
        self.conv3.weight = params['conv3_weight']
        self.conv3.bias = params['conv3_bias']
        self.conv4.weight = params['conv4_weight']
        self.conv4.bias = params['conv4_bias']
        if 'fc1_weight' in params:
            # 如果参数中有fc层，说明已经初始化过
            if not self._fc_initialized:
                # 从权重形状推断输入特征数
                fc1_in_features = params['fc1_weight'].shape[1]
                self.fc1 = Linear(in_features=fc1_in_features, out_features=512)
                self.fc2 = Linear(in_features=512, out_features=128)
                self.fc3 = Linear(in_features=128, out_features=1)
                self._fc_initialized = True
            self.fc1.weight = params['fc1_weight']
            self.fc1.bias = params['fc1_bias']
            self.fc2.weight = params['fc2_weight']
            self.fc2.bias = params['fc2_bias']
            self.fc3.weight = params['fc3_weight']
            self.fc3.bias = params['fc3_bias']

