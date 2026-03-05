import time
import matplotlib.pyplot as plt
import numpy as np
from dijkstra_solver import GraphTools
from experiment_utils import ExperimentUtils

def run_project2_experiment():
    print("========== Project 2: Dijkstra vs LP Experiment ==========")
    
    # 实验配置
    node_counts = [10, 30, 50, 100, 200, 300]
    num_trials = 5 # 每个规模重复实验次数取平均
    
    dijkstra_times = []
    lp_times = []
    
    for n in node_counts:
        print(f"Testing graph size: N={n} ...")
        
        d_time_sum = 0
        lp_time_sum = 0
        valid_trials = 0
        
        for _ in range(num_trials):
            # 1. 生成图
            adj_list, _ = ExperimentUtils.generate_random_connected_graph(n)
            
            # 检查连通性 (Task requirement)
            if not GraphTools.check_connectivity(n, adj_list):
                print("  Skipping disconnected graph...")
                continue
            
            # 检查负权重 (Task requirement)
            if GraphTools.check_negative_weights(adj_list):
                 print("  Skipping graph with negative weights...")
                 continue

            start_node = 0
            end_node = n - 1
            
            # 2. Benchmark Dijkstra
            t0 = time.time()
            dist_dijkstra, _ = GraphTools.dijkstra(n, adj_list, start_node, end_node)
            t1 = time.time()
            d_time_sum += (t1 - t0)
            
            # 3. Benchmark LP
            # 注意: LP 求解器内部会计时，这里我们记录函数返回的时间
            cost_lp, lp_duration = ExperimentUtils.solve_shortest_path_lp(n, adj_list, start_node, end_node)
            lp_time_sum += lp_duration
            
            # 简单验证结果一致性 (允许浮点误差)
            if abs(dist_dijkstra - cost_lp) > 1e-4:
                print(f"  Warning: Mismatch! Dijkstra={dist_dijkstra}, LP={cost_lp}")
            
            valid_trials += 1
            
        if valid_trials > 0:
            dijkstra_times.append(d_time_sum / valid_trials)
            lp_times.append(lp_time_sum / valid_trials)
        else:
            dijkstra_times.append(0)
            lp_times.append(0)
            
    # 4. 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, dijkstra_times, 'o-', label='Dijkstra (Heapq)')
    plt.plot(node_counts, lp_times, 's-', label='Linear Programming (SciPy Highs)')
    
    plt.title('Performance Comparison: Dijkstra vs LP Solver')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    output_path = 'dijkstra_vs_lp_performance.png'
    plt.savefig(output_path)
    print(f"Result plot saved to {output_path}")
    print("Project 2 Done.")

if __name__ == "__main__":
    run_project2_experiment()
