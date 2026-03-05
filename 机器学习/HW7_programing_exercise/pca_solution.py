import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os

# Set up paths
data_dir = os.path.join(os.getcwd(), 'hw7_data')
output_dir = os.path.join(os.getcwd(), 'results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

categories = ['guitar', 'tomato', 'tree']

for cat in categories:
    print(f"Processing category: {cat}")
    
    # 1. Data Preprocessing
    file_path = os.path.join(data_dir, f'{cat}.parquet')
    df = pd.read_parquet(file_path)
    
    # The data is in a column 'images' which contains lists of 4096 elements
    X = np.stack(df['images'].values).T
    
    # Ensure X is d x n (4096 x 60)
    d, n = X.shape
    print(f"  Data shape (d x n): {d} x {n}")
    
    # Compute sample mean
    x_bar = np.mean(X, axis=1, keepdims=True)
    
    # Create centered data matrix
    X_tilde = X - x_bar
    
    # 2. Principal component computation
    # X_tilde = U * Sigma * V^T
    # full_matrices=False gives us U of shape (d, n) if d > n
    U, s, Vt = svd(X_tilde, full_matrices=False)
    
    # Singular values are the diagonal entries of Sigma
    # u1, u2 are the first two columns of U
    u1 = U[:, 0]
    u2 = U[:, 1]
    
    # Reshape and display x_bar, u1, u2 into 64x64 images
    images_to_plot = {
        'Sample Mean': x_bar.reshape(64, 64),
        'PC1 (u1)': u1.reshape(64, 64),
        'PC2 (u2)': u2.reshape(64, 64)
    }
    
    plt.figure(figsize=(15, 5))
    for i, (title, img) in enumerate(images_to_plot.items()):
        plt.subplot(1, 3, i + 1)
        # Use grayscale
        plt.imshow(img, cmap='gray')
        plt.title(f"{cat.capitalize()} - {title}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{cat}_pca_vis.png'))
    plt.close()
    
    # Plot singular values
    plt.figure(figsize=(8, 4))
    plt.stem(range(1, len(s) + 1), s)
    plt.title(f"{cat.capitalize()} - Singular Values")
    plt.xlabel("Component Index")
    plt.ylabel("Singular Value ($\sigma_i$)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{cat}_singular_values.png'))
    plt.close()

print("Processing complete. Results saved in 'results' folder.")
