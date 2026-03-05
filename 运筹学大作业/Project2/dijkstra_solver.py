import heapq
import collections

class GraphTools:
    """
    Project 2 要求的图算法工具类
    包含：
    1. 连通性检查
    2. 负权重边检查
    3. Dijkstra 最短路算法 (使用 heapq)
    """

    @staticmethod
    def check_connectivity(num_nodes, adj_list):
        """
        检查图是否连通 (使用 BFS)
        :param num_nodes: 节点数量
        :param adj_list: 邻接表 {u: [(v, weight), ...]}
        :return: bool
        """
        if num_nodes == 0:
            return True
        
        start_node = 0
        visited = set()
        queue = collections.deque([start_node])
        visited.add(start_node)
        
        while queue:
            u = queue.popleft()
            if u in adj_list:
                for v, _ in adj_list[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
        
        return len(visited) == num_nodes

    @staticmethod
    def check_negative_weights(adj_list):
        """
        检查是否存在负权重边
        :param adj_list: 邻接表
        :return: bool (True if negative edge exists)
        """
        for u in adj_list:
            for v, weight in adj_list[u]:
                if weight < 0:
                    return True
        return False

    @staticmethod
    def dijkstra(num_nodes, adj_list, start_node, end_node):
        """
        Dijkstra 算法实现 (使用 heapq)
        :param num_nodes: 节点数量
        :param adj_list: 邻接表 {u: [(v, weight), ...]}
        :param start_node: 起点
        :param end_node: 终点
        :return: (shortest_distance, path)
        """
        # 初始化距离字典，float('inf') 代表无穷大
        distances = {node: float('inf') for node in range(num_nodes)}
        distances[start_node] = 0
        
        # 前驱节点记录，用于还原路径
        predecessors = {node: None for node in range(num_nodes)}
        
        # 优先队列 (min-heap), 存储 (current_dist, u)
        # 必须使用 heapq，不能用 PriorityQueue
        pq = [(0, start_node)]
        
        shortest_path_found = False
        
        while pq:
            d, u = heapq.heappop(pq)
            
            # 如果当前距离已经大于已知最短距离，跳过 (Lazy Deletion)
            if d > distances[u]:
                continue
            
            # 找到终点
            if u == end_node:
                shortest_path_found = True
                break
            
            if u in adj_list:
                for v, weight in adj_list[u]:
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u
                        heapq.heappush(pq, (distances[v], v))
                        
        if distances[end_node] == float('inf'):
            return float('inf'), []
            
        # 还原路径
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()
        
        return distances[end_node], path
