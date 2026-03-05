"""
逻辑回归分类器 - MNIST手写数字识别（二分类：判断是否为数字6）
实现练习7中的所有要求
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import gzip
from typing import Tuple, Dict, List
from collections import defaultdict


class LogisticRegression:
    """逻辑回归分类器"""
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 2000):
        """
        初始化逻辑回归分类器
        
        Args:
            learning_rate: 学习率
            max_iter: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.train_accuracies = []
        self.iterations = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        # 为了数值稳定性，限制z的范围
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        计算逻辑回归的损失函数
        
        L(w) = -1/n * Σ[y_i * log(h(x_i)) + (1-y_i) * log(1-h(x_i))]
        """
        m = X.shape[0]
        h = self.sigmoid(X @ w)
        # 添加小常数避免log(0)
        epsilon = 1e-10
        loss = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return loss
    
    def compute_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        计算梯度
        
        ∇L(w) = 1/n * X^T * (h(X@w) - y)
        """
        m = X.shape[0]
        h = self.sigmoid(X @ w)
        gradient = 1/m * X.T @ (h - y)
        return gradient
    
    def fit_gd(self, X_train: np.ndarray, y_train: np.ndarray, 
               X_val: np.ndarray = None, y_val: np.ndarray = None,
               early_stop_acc: float = 0.95) -> Dict:
        """
        使用梯度下降(GD)训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            early_stop_acc: 早停准确率阈值
            
        Returns:
            训练历史字典
        """
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        
        self.train_accuracies = []
        self.iterations = []
        
        start_time = time.time()
        
        for i in range(self.max_iter):
            # 计算梯度
            gradient = self.compute_gradient(X_train, y_train, self.w)
            
            # 更新参数
            self.w -= self.learning_rate * gradient
            
            # 每10次迭代记录一次准确率
            if i % 10 == 0 or i == self.max_iter - 1:
                if X_val is not None:
                    acc = self.evaluate(X_val, y_val)
                else:
                    acc = self.evaluate(X_train, y_train)
                
                self.train_accuracies.append(acc)
                self.iterations.append(i)
                
                if i % 100 == 0:
                    print(f"Iteration {i}: Accuracy = {acc:.4f}")
                
                # 早停条件
                if acc >= early_stop_acc:
                    print(f"Early stopping at iteration {i} with accuracy {acc:.4f}")
                    break
        
        train_time = time.time() - start_time
        final_iter = self.iterations[-1]
        
        return {
            'method': 'GD',
            'iterations': final_iter,
            'time': train_time,
            'final_accuracy': self.train_accuracies[-1],
            'accuracy_history': self.train_accuracies.copy(),
            'iteration_history': self.iterations.copy()
        }
    
    def fit_sgd(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                early_stop_acc: float = 0.95,
                sampling_strategy: str = 'with_replacement') -> Dict:
        """
        使用随机梯度下降(SGD)训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            early_stop_acc: 早停准确率阈值
            sampling_strategy: 采样策略
                - 'without_replacement': 无放回随机采样（每个epoch打乱一次）
                - 'with_replacement': 有放回随机采样（每次迭代随机选一个样本）
                - 'mini_batch': Mini-batch采样
                
        Returns:
            训练历史字典
        """
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        
        self.train_accuracies = []
        self.iterations = []
        
        start_time = time.time()
        iteration = 0
        
        if sampling_strategy == 'without_replacement':
            # 无放回采样：每个epoch打乱数据
            for epoch in range(self.max_iter // n_samples + 1):
                # 打乱数据
                indices = np.random.permutation(n_samples)
                
                for idx in indices:
                    # 单个样本的梯度
                    x_i = X_train[idx:idx+1]
                    y_i = y_train[idx:idx+1]
                    
                    h = self.sigmoid(x_i @ self.w)
                    gradient = x_i.T @ (h - y_i)
                    gradient = gradient.flatten()
                    
                    # 更新参数
                    self.w -= self.learning_rate * gradient
                    
                    iteration += 1
                    
                    # 每10次迭代记录一次
                    if iteration % 10 == 0 or iteration >= self.max_iter:
                        if X_val is not None:
                            acc = self.evaluate(X_val, y_val)
                        else:
                            acc = self.evaluate(X_train, y_train)
                        
                        self.train_accuracies.append(acc)
                        self.iterations.append(iteration)
                        
                        if iteration % 100 == 0:
                            print(f"Iteration {iteration}: Accuracy = {acc:.4f}")
                        
                        # 早停
                        if acc >= early_stop_acc or iteration >= self.max_iter:
                            if acc >= early_stop_acc:
                                print(f"Early stopping at iteration {iteration} with accuracy {acc:.4f}")
                            break
                
                if iteration >= self.max_iter or (self.train_accuracies and self.train_accuracies[-1] >= early_stop_acc):
                    break
                    
        elif sampling_strategy == 'with_replacement':
            # 有放回采样：每次迭代随机选择一个样本
            for iteration in range(self.max_iter):
                # 随机选择一个样本
                idx = np.random.randint(n_samples)
                
                x_i = X_train[idx:idx+1]
                y_i = y_train[idx:idx+1]
                
                h = self.sigmoid(x_i @ self.w)
                gradient = x_i.T @ (h - y_i)
                gradient = gradient.flatten()
                
                # 更新参数
                self.w -= self.learning_rate * gradient
                
                # 每10次迭代记录一次
                if iteration % 10 == 0 or iteration == self.max_iter - 1:
                    if X_val is not None:
                        acc = self.evaluate(X_val, y_val)
                    else:
                        acc = self.evaluate(X_train, y_train)
                    
                    self.train_accuracies.append(acc)
                    self.iterations.append(iteration)
                    
                    if iteration % 100 == 0:
                        print(f"Iteration {iteration}: Accuracy = {acc:.4f}")
                    
                    # 早停
                    if acc >= early_stop_acc:
                        print(f"Early stopping at iteration {iteration} with accuracy {acc:.4f}")
                        break
                        
        elif sampling_strategy == 'mini_batch':
            # Mini-batch采样
            batch_size = 32
            n_batches = n_samples // batch_size
            
            for epoch in range(self.max_iter // n_batches + 1):
                # 打乱数据
                indices = np.random.permutation(n_samples)
                
                for batch_idx in range(n_batches):
                    # 获取batch
                    batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]
                    
                    # 计算batch梯度
                    gradient = self.compute_gradient(X_batch, y_batch, self.w)
                    
                    # 更新参数
                    self.w -= self.learning_rate * gradient
                    
                    iteration += 1
                    
                    # 每10次迭代记录一次
                    if iteration % 10 == 0 or iteration >= self.max_iter:
                        if X_val is not None:
                            acc = self.evaluate(X_val, y_val)
                        else:
                            acc = self.evaluate(X_train, y_train)
                        
                        self.train_accuracies.append(acc)
                        self.iterations.append(iteration)
                        
                        if iteration % 100 == 0:
                            print(f"Iteration {iteration}: Accuracy = {acc:.4f}")
                        
                        # 早停
                        if acc >= early_stop_acc or iteration >= self.max_iter:
                            if acc >= early_stop_acc:
                                print(f"Early stopping at iteration {iteration} with accuracy {acc:.4f}")
                            break
                
                if iteration >= self.max_iter or (self.train_accuracies and self.train_accuracies[-1] >= early_stop_acc):
                    break
        
        train_time = time.time() - start_time
        final_iter = self.iterations[-1] if self.iterations else 0
        
        return {
            'method': f'SGD ({sampling_strategy})',
            'iterations': final_iter,
            'time': train_time,
            'final_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'accuracy_history': self.train_accuracies.copy(),
            'iteration_history': self.iterations.copy()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签"""
        probabilities = self.sigmoid(X @ self.w)
        return (probabilities >= 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def compute_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """计算混淆矩阵"""
        predictions = self.predict(X)
        
        tp = np.sum((y == 1) & (predictions == 1))
        tn = np.sum((y == 0) & (predictions == 0))
        fp = np.sum((y == 0) & (predictions == 1))
        fn = np.sum((y == 1) & (predictions == 0))
        
        return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}
    
    def compute_metrics(self, cm: Dict) -> Dict:
        """计算评估指标"""
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }


def read_mnist_images(filename: str) -> np.ndarray:
    """读取MNIST图像文件"""
    with gzip.open(filename, 'rb') as f:
        # 读取magic number和维度信息
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # 读取图像数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images


def read_mnist_labels(filename: str) -> np.ndarray:
    """读取MNIST标签文件"""
    with gzip.open(filename, 'rb') as f:
        # 读取magic number和数量
        magic, num = struct.unpack(">II", f.read(8))
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist_data(target_digit: int = 6) -> Tuple:
    """
    加载MNIST数据并转换为二分类问题
    
    Args:
        target_digit: 目标数字（默认为6）
        
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    print("加载MNIST数据集...")
    
    # 数据文件路径
    train_images_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
    train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
    test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
    test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
    
    # 检查文件是否存在，如果不存在则尝试读取未压缩版本
    import os
    if not os.path.exists(train_images_path):
        train_images_path = train_images_path.replace('.gz', '')
        train_labels_path = train_labels_path.replace('.gz', '')
        test_images_path = test_images_path.replace('.gz', '')
        test_labels_path = test_labels_path.replace('.gz', '')
        
        # 如果是未压缩文件，使用不同的读取方法
        def read_mnist_images_uncompressed(filename):
            with open(filename, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(num, rows * cols)
            return images
        
        def read_mnist_labels_uncompressed(filename):
            with open(filename, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        
        X_train = read_mnist_images_uncompressed(train_images_path).astype(np.float64)
        y_train = read_mnist_labels_uncompressed(train_labels_path)
        X_test = read_mnist_images_uncompressed(test_images_path).astype(np.float64)
        y_test = read_mnist_labels_uncompressed(test_labels_path)
    else:
        # 读取压缩文件
        X_train = read_mnist_images(train_images_path).astype(np.float64)
        y_train = read_mnist_labels(train_labels_path)
        X_test = read_mnist_images(test_images_path).astype(np.float64)
        y_test = read_mnist_labels(test_labels_path)
    
    # 转换为二分类：是否为目标数字
    y_train = (y_train == target_digit).astype(int)
    y_test = (y_test == target_digit).astype(int)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"目标数字 {target_digit} 在训练集中的比例: {np.mean(y_train):.4f}")
    print(f"目标数字 {target_digit} 在测试集中的比例: {np.mean(y_test):.4f}")
    
    return X_train, y_train, X_test, y_test


def normalize_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    归一化数据：将像素值从[0, 255]缩放到[0, 1]
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        
    Returns:
        (X_train_normalized, X_test_normalized)
    """
    print("\n归一化数据...")
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    
    # 添加偏置项（intercept）
    X_train_norm = np.c_[np.ones(X_train_norm.shape[0]), X_train_norm]
    X_test_norm = np.c_[np.ones(X_test_norm.shape[0]), X_test_norm]
    
    print(f"归一化后的数据范围: [{X_train_norm[:, 1:].min():.2f}, {X_train_norm[:, 1:].max():.2f}]")
    print(f"特征维度（含偏置项）: {X_train_norm.shape[1]}")
    
    return X_train_norm, X_test_norm


def compute_lipschitz_constant(X: np.ndarray) -> float:
    """
    计算目标函数∇L(w)的Lipschitz常数
    
    对于逻辑回归，Lipschitz常数 L = λ_max(X^T X) / (4n)
    其中λ_max是X^T X的最大特征值
    
    Args:
        X: 数据矩阵
        
    Returns:
        Lipschitz常数
    """
    print("\n计算Lipschitz常数...")
    n = X.shape[0]
    
    # 计算X^T X的最大特征值
    # 由于X^T X可能很大，使用幂迭代法估计最大特征值
    XTX = X.T @ X
    
    # 使用numpy的特征值计算（对于小矩阵）或幂迭代法
    if X.shape[1] <= 1000:
        eigenvalues = np.linalg.eigvalsh(XTX)
        lambda_max = np.max(eigenvalues)
    else:
        # 幂迭代法
        v = np.random.randn(X.shape[1])
        v = v / np.linalg.norm(v)
        
        for _ in range(100):
            v_new = XTX @ v
            v_new = v_new / np.linalg.norm(v_new)
            if np.allclose(v, v_new):
                break
            v = v_new
        
        lambda_max = v.T @ XTX @ v
    
    L = lambda_max / (4 * n)
    
    print(f"X^T X的最大特征值: {lambda_max:.4f}")
    print(f"Lipschitz常数 L: {L:.6f}")
    print(f"建议的学习率上界: {1/L:.6f}")
    
    return L


def plot_accuracy_curves(results_list: List[Dict], title: str = "Training Accuracy vs Iterations"):
    """绘制准确率曲线"""
    plt.figure(figsize=(12, 6))
    
    for result in results_list:
        plt.plot(result['iteration_history'], 
                result['accuracy_history'], 
                label=result['method'], 
                marker='o', 
                markersize=3,
                linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    filename = title.replace(' ', '_').replace(':', '').lower() + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存为: {filename}")
    plt.show()


def print_results(result: Dict, cm: Dict, metrics: Dict):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"方法: {result['method']}")
    print(f"{'='*60}")
    print(f"训练时间: {result['time']:.2f} 秒")
    print(f"迭代次数: {result['iterations']}")
    print(f"最终准确率: {result['final_accuracy']:.4f}")
    
    print(f"\n混淆矩阵:")
    print(f"{'':>15} {'预测为6':>15} {'预测非6':>15}")
    print(f"{'实际为6':>15} {cm['TP']:>15} {cm['FN']:>15}")
    print(f"{'实际非6':>15} {cm['FP']:>15} {cm['TN']:>15}")
    
    print(f"\n评估指标:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision): {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"  F1分数 (F1-Score):  {metrics['f1_score']:.4f}")


def main():
    """主函数"""
    print("="*60)
    print("逻辑回归手写数字分类器 - MNIST数据集")
    print("任务: 判断手写数字是否为6")
    print("="*60)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_data(target_digit=6)
    
    # 任务1: 归一化数据
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # 任务1: 计算Lipschitz常数
    lipschitz_const = compute_lipschitz_constant(X_train_norm)
    
    # 根据Lipschitz常数设置学习率
    suggested_lr = 1 / lipschitz_const
    learning_rate = min(0.1, suggested_lr * 0.5)  # 使用一个保守的学习率
    print(f"\n使用的学习率: {learning_rate:.6f}")
    
    # ========== 任务2(a): GD vs SGD ==========
    print("\n" + "="*60)
    print("任务2(a): 使用梯度下降(GD)和随机梯度下降(SGD)训练")
    print("="*60)
    
    # 梯度下降
    print("\n训练梯度下降(GD)模型...")
    print("-"*60)
    lr_gd = LogisticRegression(learning_rate=learning_rate, max_iter=2000)
    result_gd = lr_gd.fit_gd(X_train_norm, y_train, X_train_norm, y_train, early_stop_acc=0.95)
    
    # 随机梯度下降（无放回采样）
    print("\n训练随机梯度下降(SGD)模型 - 无放回采样...")
    print("-"*60)
    lr_sgd = LogisticRegression(learning_rate=learning_rate, max_iter=2000)
    result_sgd = lr_sgd.fit_sgd(X_train_norm, y_train, X_train_norm, y_train, 
                               early_stop_acc=0.95, sampling_strategy='without_replacement')
    
    # 绘制准确率曲线
    plot_accuracy_curves([result_gd, result_sgd], 
                         title="Task 2(a): GD vs SGD - Accuracy vs Iterations")
    
    # ========== 任务2(b): 比较迭代次数和时间 ==========
    print("\n" + "="*60)
    print("任务2(b): 比较GD和SGD的效率")
    print("="*60)
    print(f"\n{'方法':<20} {'迭代次数':<15} {'训练时间(秒)':<20} {'最终准确率':<15}")
    print("-"*60)
    print(f"{result_gd['method']:<20} {result_gd['iterations']:<15} {result_gd['time']:<20.2f} {result_gd['final_accuracy']:<15.4f}")
    print(f"{result_sgd['method']:<20} {result_sgd['iterations']:<15} {result_sgd['time']:<20.2f} {result_sgd['final_accuracy']:<15.4f}")
    
    # ========== 任务2(c): 比较分类性能 ==========
    print("\n" + "="*60)
    print("任务2(c): 在测试集上比较GD和SGD的性能")
    print("="*60)
    
    # GD结果
    cm_gd = lr_gd.compute_confusion_matrix(X_test_norm, y_test)
    metrics_gd = lr_gd.compute_metrics(cm_gd)
    print_results(result_gd, cm_gd, metrics_gd)
    
    # SGD结果
    cm_sgd = lr_sgd.compute_confusion_matrix(X_test_norm, y_test)
    metrics_sgd = lr_sgd.compute_metrics(cm_sgd)
    print_results(result_sgd, cm_sgd, metrics_sgd)
    
    # ========== 任务3: 不同采样策略 ==========
    print("\n" + "="*60)
    print("任务3: 比较不同采样策略")
    print("="*60)
    
    sampling_results = []
    
    # 无放回采样（已经训练过了）
    sampling_results.append(result_sgd)
    
    # 有放回采样
    print("\n训练SGD模型 - 有放回采样...")
    print("-"*60)
    lr_sgd_with = LogisticRegression(learning_rate=learning_rate, max_iter=2000)
    result_sgd_with = lr_sgd_with.fit_sgd(X_train_norm, y_train, X_train_norm, y_train,
                                         early_stop_acc=0.95, sampling_strategy='with_replacement')
    sampling_results.append(result_sgd_with)
    
    # Mini-batch采样
    print("\n训练SGD模型 - Mini-batch采样...")
    print("-"*60)
    lr_sgd_mini = LogisticRegression(learning_rate=learning_rate, max_iter=2000)
    result_sgd_mini = lr_sgd_mini.fit_sgd(X_train_norm, y_train, X_train_norm, y_train,
                                         early_stop_acc=0.95, sampling_strategy='mini_batch')
    sampling_results.append(result_sgd_mini)
    
    # 绘制比较图
    plot_accuracy_curves(sampling_results, 
                        title="Task 3: Sampling Strategies Comparison")
    
    # 任务3(b): 比较收敛速度和稳定性
    print("\n" + "="*60)
    print("任务3(b): 采样策略性能对比")
    print("="*60)
    print(f"\n{'采样策略':<30} {'迭代次数':<15} {'训练时间(秒)':<20} {'最终准确率':<15}")
    print("-"*60)
    for result in sampling_results:
        print(f"{result['method']:<30} {result['iterations']:<15} {result['time']:<20.2f} {result['final_accuracy']:<15.4f}")
    
    print("\n收敛速度和稳定性分析:")
    print("-"*60)
    for result in sampling_results:
        if len(result['accuracy_history']) > 1:
            # 计算准确率的标准差（作为稳定性指标）
            acc_std = np.std(result['accuracy_history'][-10:]) if len(result['accuracy_history']) >= 10 else 0
            print(f"{result['method']:<30} 最后10次迭代的准确率标准差: {acc_std:.6f}")
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == '__main__':
    main()
