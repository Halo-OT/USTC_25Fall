import os
from sklearn.datasets import load_svmlight_file
import numpy as np

def get_a9a_data(data_dir="."):
    """
    加载本地 a9a 数据集 (libsvm格式)。
    :param data_dir: 数据文件所在目录
    :return: (X, y)
    """
    # 尝试在 data_dir 或其父目录查找 a9a.txt
    potential_paths = [
        os.path.join(data_dir, "a9a.txt"),
        os.path.join(data_dir, "data", "a9a.txt"),
        os.path.join(os.path.dirname(data_dir), "a9a.txt"),
        "a9a.txt"
    ]
    
    file_path = None
    for p in potential_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        raise FileNotFoundError(f"Could not find a9a.txt in searched paths: {potential_paths}")

    print(f"Loading data from {file_path} ...")
    
    try:
        # a9a has 123 features. 
        # Using n_features=123 ensures consistent shape even if last feature is sparse in subset
        X, y = load_svmlight_file(file_path, n_features=123)
        
        # 转换为 dense array 以便处理
        X = X.toarray()
        
        # 增加 bias 列 (intercept) - 全 1 的列
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # 确保标签是 -1, 1
        # a9a 原始标签通常是 -1, +1. 如果不是，这里通过 unique 判断并转换
        # a9a data format: -1 or +1
        # scikit-learn load_svmlight_file might return labels as floats
        
        y = y.reshape(-1) # ensure 1D
        unique_labels = np.unique(y)
        # 如果标签是 0/1，转为 -1/1
        if set(unique_labels) == {0, 1}:
            y = np.where(y == 0, -1, 1)
            
        return X, y
        
    except Exception as e:
        print(f"Error loading a9a: {e}")
        raise e

