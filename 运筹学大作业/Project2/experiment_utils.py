import networkx as nx
import numpy as np
from scipy.optimize import linprog
import random
import time

class ExperimentUtils:
    """
    Project 2 实验辅助工具
    包含：
    1. 随机图生成 (ER图)
    2. LP 模型构建与求解
    """

    @staticmethod
    def generate_random_connected_graph(n, p=None, seed=None):
        """
        生成随机连通图 (Erdős–Rényi graph)
        保证图是连通的，并且边权重为正。
        :param n: 节点数量
        :param p: 连接概率. 如果为 None，自动设定以保证连通性.
        :param seed: 随机种子
        :return: (adj_list, networkx_graph)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # 根据阈值定理设定 p，保证连通性
        # Threshold is ln(n)/n. We employ a strictly larger p.
        if p is None:
            if n <= 1:
                p = 1.0
            else:
                p = (1.5 * np.log(n)) / n
                p = min(p, 1.0) # Cap at 1.0

        connected = False
        G = None
        
        # 循环直到生成连通图
        while not connected:
            G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
            connected = nx.is_strongly_connected(G)
            if not connected and seed is not None:
                 print("Warning: Fixed seed produced disconnected graph, retrying without fixed seed...")
                 seed = None # 如果指定种子导致不连通，后续尝试随机
        
        # 为边分配随机正权重
        adj_list = {i: [] for i in range(n)}
        for u, v in G.edges():
            weight = random.uniform(1, 10) # 权重 1 到 10
            G[u][v]['weight'] = weight
            adj_list[u].append((v, weight))
            
        return adj_list, G

    @staticmethod
    def solve_shortest_path_lp(num_nodes, adj_list, start_node, end_node):
        """
        使用线性规划 (LP) 求解最短路
        Model:
        min sum(c_ij * x_ij)
        s.t. sum(x_ki) - sum(x_ik) = b_k  (Flow conservation)
             x_ij >= 0
             
        b_k = 1 if k=start, -1 if k=end, 0 otherwise
        
        :return: (optimal_cost, time_taken)
        """
        # 1. 整理所有的边，建立 边 -> index 的映射
        edges = []
        for u in adj_list:
            for v, w in adj_list[u]:
                edges.append((u, v, w))
        
        num_edges = len(edges)
        if num_edges == 0:
            return float('inf'), 0.0

        # 2. 构建目标函数系数 c
        c = np.array([w for _, _, w in edges])

        # 3. 构建约束矩阵 A_eq 和 b_eq
        # A_eq 维度: (num_nodes, num_edges)
        # 每一行代表一个节点 k 的流守恒约束
        A_eq = np.zeros((num_nodes, num_edges))
        
        for edge_idx, (u, v, _) in enumerate(edges):
            # 流出 u: -1 (或者 +1, 只要一致即可，通常流出为+则流入为-)
            # 这里定义： 入流 - 出流 = 需求
            # sum(x_in) - sum(x_out) = b_k
            # x_uv 是一条从 u 到 v 的边
            # 对节点 u: x_uv 是出流 -> 系数 -1
            # 对节点 v: x_uv 是入流 -> 系数 +1
            A_eq[u, edge_idx] = -1
            A_eq[v, edge_idx] = 1

        b_eq = np.zeros(num_nodes)
        b_eq[start_node] = -1 # 源点产生流量，净流出为1 => (入-出) = -1
        b_eq[end_node] = 1    # 汇点接收流量，净流入为1 => (入-出) = 1
        
        # 4. 求解
        start_time = time.time()
        # method='highs' 是 scipy 推荐的高性能求解器
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        end_time = time.time()
        
        if res.success:
            return res.fun, end_time - start_time
        else:
            return float('inf'), end_time - start_time
