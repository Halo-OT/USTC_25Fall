import numpy as np
import scipy.sparse

class LogisticRegressionModel:
    """
    Project 3: Logistic Regression with L2 Regularization
    Target Function:
    min L(x) = (1/m) * sum( ln(1 + exp(-b_i * a_i^T * x)) ) + lambda * ||x||^2
    """
    def __init__(self, A, b, lambda_reg=1e-5):
        """
        初始化模型
        :param A: 样本特征矩阵 (m, n) (可以是稀疏矩阵 或 numpy array)
        :param b: 标签向量 (m,) , 值为 +1 或 -1
        :param lambda_reg: 正则化参数
        """
        self.A = A
        self.b = b
        self.m, self.n = A.shape
        self.lambda_reg = lambda_reg
        
        # 确保 b 是 (m, 1) 的形状以便广播
        self.b = self.b.reshape(-1, 1)

    def value(self, x):
        """
        计算函数值 L(x)
        :param x: 参数向量 (n, 1)
        :return: scalar value
        """
        # 计算 z = - b * (A x)
        # 注意: 稀疏矩阵乘法 A.dot(x)
        Ax = self.A.dot(x)
        z = - self.b * Ax
        
        # 计算 loss term: mean( ln(1 + exp(z)) )
        # 使用 numpy.logaddexp(0, z) 来数值稳定地计算 ln(1 + exp(z))
        # logaddexp(x, y) = log(exp(x) + exp(y))
        # 这里我们需要 log(1 + exp(z)) = log(exp(0) + exp(z))
        loss_term = np.mean(np.logaddexp(0, z))
        
        # 计算 regularization term: lambda * ||x||^2
        reg_term = self.lambda_reg * np.sum(x**2)
        
        return loss_term + reg_term

    def gradient(self, x):
        """
        计算梯度 gradient
        grad = (1/m) * A^T * (b * (sigmoid(-b*A*x) - 1)) + 2*lambda*x
               推导变化：
               d/dx ln(1+exp(-yi wTx)) = ... = -yi * sigmoid(-yi wTx) * xi
                                       = -yi * (1 / (1 + exp(yi wTx))) * xi
               文档公式为：
               grad = (1/m) * sum( -b_i * a_i / (1 + exp(b_i * a_i^T * x)) ) + 2*lambda*x
               
               令 p_i = 1 / (1 + exp(b_i * a_i^T * x)) = sigmoid(-z) 其中 z_i = -b_i a_i^T x
               grad = (1/m) * sum( -b_i * p_i * a_i ) + 2*lambda*x
               Vectorized:
               coeff = -b * p  (element-wise)
               grad = (1/m) * A^T * coeff + 2*lambda*x
        """
        Ax = self.A.dot(x)
        # exp_arg = b_i * a_i^T * x
        exp_arg = self.b * Ax
        
        # p = 1 / (1 + exp(exp_arg))
        # 使用 scipy.special.expit(x) = sigmoid(x) = 1/(1+exp(-x))
        # 我们要计算 1/(1+exp(u)) = sigmoid(-u)
        from scipy.special import expit
        p = expit(-exp_arg)
        
        # coeff = -b * p
        coeff = -self.b * p
        
        # grad = (1/m) * A.T * coeff + 2 * lambda * x
        # Dealing with sparse matrix A
        if scipy.sparse.issparse(self.A):
            grad_data = self.A.T.dot(coeff)
        else:
            grad_data = self.A.T @ coeff
            
        grad = (1/self.m) * grad_data + 2 * self.lambda_reg * x
        return grad

    def hessian(self, x):
        """
        计算 Hessian 矩阵
        H = (1/m) * A^T * D * A + 2*lambda*I
        其中 D 是对角阵，D_ii = p_i * (1 - p_i)
        p_i = sigmoid(- b_i * a_i^T * x)
        注意：如果 n 很大，直接存储 Hessian (n, n) 可能会爆内存。
        对于 a9a (n ~ 123), 可以直接存储。
        """
        Ax = self.A.dot(x)
        exp_arg = self.b * Ax
        
        from scipy.special import expit
        p = expit(-exp_arg) # p is (m, 1)
        
        # d vector for diagonal D
        d = p * (1 - p) # (m, 1)
        
        # H_part1 = A.T @ diag(d) @ A
        # 高效计算: A.T @ (d * A) 利用广播
        # 但如果是稀疏矩阵:
        if scipy.sparse.issparse(self.A):
            # 构造稀疏对角阵 D
            D = scipy.sparse.diags(d.flatten())
            H_part1 = self.A.T.dot(D).dot(self.A).toarray() # 转换回 dense，因为 n 不大
        else:
            H_part1 = (self.A.T * d.T) @ self.A
            
        H = (1/self.m) * H_part1 + 2 * self.lambda_reg * np.eye(self.n)
        return H
