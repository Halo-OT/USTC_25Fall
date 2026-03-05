import numpy as np

class Optimizer:
    """
    Project 3: Optimization Algorithms
    包含：
    1. Backtracking Line Search
    2. Newton's Method
    """
    
    @staticmethod
    def backtracking_line_search(model, x, d, grad, alpha0=1.0, gamma=0.5, c=1e-4):
        """
        回溯线搜索
        :param model: 目标模型，需实现 value(x)
        :param x: 当前点
        :param d: 下降方向 (Newton direction)
        :param grad: 当前点的梯度
        :param alpha0: 初始步长
        :param gamma: 缩减因子
        :param c: Armijo 条件参数
        :return: 合适的步长 alpha
        """
        alpha = alpha0
        f_curr = model.value(x)
        grad_dot_d = np.dot(grad.T, d).item() # scalar
        
        while True:
            x_new = x + alpha * d
            f_new = model.value(x_new)
            
            # Armijo Condition: f(x + alpha*d) <= f(x) + c * alpha * grad^T * d
            if f_new <= f_curr + c * alpha * grad_dot_d:
                break
            
            alpha *= gamma
            
            # 安全中断，避免死循环
            if alpha < 1e-10:
                break
                
        return alpha

    @staticmethod
    def newton_method(model, x0, max_iter=100, tol=1e-6, ls_gamma=0.5, ls_c=1e-4):
        """
        牛顿法主循环
        :param model: 需实现 gradient(x) 和 hessian(x)
        :param x0: 初始点
        :param ls_gamma: 线搜索缩减因子
        :param ls_c: Armijo 条件参数
        :return: (x_opt, path_info)
        """
        x = x0.copy()
        
        # 记录收敛信息：iterations, loss_diff, grad_norm
        history = {
            'loss': [],
            'grad_norm': [],
            'step_size': []
        }
        
        print(f"{'Iter':<5} | {'Loss':<12} | {'Grad Norm':<12} | {'Alpha':<10}")
        print("-" * 50)
        
        for k in range(max_iter):
            loss = model.value(x)
            grad = model.gradient(x)
            grad_norm = np.linalg.norm(grad)
            hess = model.hessian(x)
            
            history['loss'].append(loss)
            history['grad_norm'].append(grad_norm)
            
            if grad_norm < tol:
                print(f"Converged at iter {k}")
                break
            
            # Newton Direction: d = - H^-1 * g
            # Solve H * d = -g for d
            try:
                d = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                print("Hessian is singular, using gradient descent direction")
                d = -grad
            
            # Line Search
            # Engineering Tip: 线搜索中，每步初始步长可以延续上一个步骤的步长 (文档提及), gamma=ls_gamma, c=ls_c
            # 但牛顿法通常 alpha0 = 1 是一个很好的尝试
            alpha = Optimizer.backtracking_line_search(model, x, d, grad, alpha0=1.0, gamma=ls_gamma, c=ls_c)
            
            history['step_size'].append(alpha)
            
            if k % 10 == 0:
                print(f"{k:<5} | {loss:<12.6f} | {grad_norm:<12.6e} | {alpha:<10.4e}")
            
            x = x + alpha * d
            
        return x, history
