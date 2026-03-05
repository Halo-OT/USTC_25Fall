import numpy as np
import matplotlib.pyplot as plt
import os
from logistic import LogisticRegressionModel
from optimizer import Optimizer
from data_loader import get_a9a_data

def run_project3_experiment():
    print("========== Project 3: Logistic Regression Optimization ==========")
    
    # 1. 加载数据
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        X, y = get_a9a_data(current_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"Data loaded: shape={X.shape}, labels={np.unique(y)}")
    
    # 2. 初始化模型
    # Regularization lambda = 1e-4 / m usually, lets pick constants from paper or default
    # Document says lambda > 0.
    model = LogisticRegressionModel(A=X, b=y, lambda_reg=1e-4)
    
    # 初始点 x0 (全0)
    x0 = np.zeros((X.shape[1], 1))
    
    # 3. 运行 Newton Method
    print("Starting Newton Method (Gamma=0.5)...")
    x_opt, history = Optimizer.newton_method(model, x0, max_iter=50, tol=1e-6, ls_gamma=0.5)
    
    # 4. 数据处理与绘图
    losses = np.array(history['loss'])
    grad_norms = np.array(history['grad_norm'])
    
    # 估计最优值 f* (取最后一次迭代的 loss，或者更精确跑更多次)
    f_star = losses[-1] 
    
    # 为了避免 log(0)，取 f_star 略小一点点或者画 difference 到倒数第二个
    # 这里我们画 f(x_k) - f^*。由于最后一个就是 f^*，差为0无法画 log，所以去掉最后一个点
    loss_diff = losses[:-1] - f_star
    # 加上一个极小值避免 zero error，或者直接切片
    loss_diff = np.maximum(loss_diff, 1e-16) 
    
    iterations = np.arange(len(loss_diff))
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Function Value Convergence
    plt.subplot(1, 2, 1)
    plt.semilogy(iterations, loss_diff, 'b-o', markersize=4)
    plt.title(r'Convergence of Function Value: $f(x_k) - f^*$')
    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x_k) - f^*$ (log scale)')
    plt.grid(True)
    
    # Plot 2: Gradient Norm Convergence
    plt.subplot(1, 2, 2)
    plt.semilogy(np.arange(len(grad_norms)), grad_norms, 'r-s', markersize=4)
    plt.title(r'Convergence of Gradient Norm: $\|\nabla f(x_k)\|$')
    plt.xlabel('Iterations')
    plt.ylabel(r'$\|\nabla f(x_k)\|$ (log scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logistic_convergence.png')
    print("Convergence plot saved to logistic_convergence.png")
    
    # 5. 额外实验：线搜索参数 Gamma 对收敛速度的影响
    print("\nStarting Parameter Sensitivity Experiment (Varying Gamma)...")
    gammas = [0.1, 0.5, 0.9]
    plt.figure(figsize=(10, 6))
    
    for g in gammas:
        print(f"Testing Gamma = {g} ...")
        _, h_g = Optimizer.newton_method(model, x0, max_iter=50, tol=1e-6, ls_gamma=g)
        
        # 处理数据以绘制 f(x) - f*
        # 注意：不同 gamma 可能收敛到的 f* 极其微小差异（数值误差），或者迭代次数不同
        # 为了对比，统一用之前求得的 f_star 作为基准
        ls_loss = np.array(h_g['loss'])
        ls_diff = ls_loss - f_star
        ls_diff = np.maximum(ls_diff, 1e-16)
        
        plt.semilogy(np.arange(len(ls_diff)), ls_diff, 'o-', label=f'Gamma={g}', markersize=3)
        
    plt.title(r'Effect of Line Search Parameter $\gamma$ on Convergence')
    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x_k) - f^*$ (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig('gamma_sensitivity.png')
    print("Sensitivity plot saved to gamma_sensitivity.png")
    
    print("Project 3 Done.")

if __name__ == "__main__":
    run_project3_experiment()
