import pandas as pd
import numpy as np

print("=" * 60)
print("PCA ON RETAIL DEMAND DATA")
print("Step-by-step from scratch")
print("=" * 60)

# ── Step 1: Load data ──
print("\n[1] Loading sales_train_validation.csv ...")
df = pd.read_csv("D:\data science\Data analysis project\Retail Demand Decomposition\Datasets\sales_train_validation.csv")
print(f"    Shape: {df.shape} (rows=items, cols=days+metadata)")

# Extract only the daily sales columns (d_1 ... d_1913)
sales_cols = [c for c in df.columns if c.startswith("d_")]
X = df[sales_cols].values
print(f"    Sales data matrix: {X.shape}")

# Use a subset for simplicity (first 30 items, all 1913 days)
n_items = 30
X_sub = X[:n_items, :]  # shape (30, 1913)
print(f"    Subset: {X_sub.shape} (30 items × 1913 days)")

# ── Step 2: Standardize ──
print("\n[2] Standardizing (mean=0, std=1) ...")
mean = X_sub.mean(axis=0)
std = X_sub.std(axis=0)
# avoid division by zero
std[std == 0] = 1.0
Z = (X_sub - mean) / std
print(f"    Standardized shape: {Z.shape}")

# ── Step 3: Covariance Matrix ──
print("\n[3] Computing covariance matrix (1913 × 1913) ...")
# Since features > observations, use: C = (1/(n-1)) * Z^T Z
C = (Z.T @ Z) / (Z.shape[0] - 1)
print(f"    Covariance matrix shape: {C.shape}")

# ── Step 4: Eigen decomposition ──
print("\n[4] Eigendecomposition ...")
eigvals, eigvecs = np.linalg.eigh(C)
# eigh returns ascending order -> flip to descending
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]
print(f"    Eigenvalues (first 5): {np.round(eigvals[:5], 4)}")
print(f"    Eigenvectors shape: {eigvecs.shape}")

# ── Step 5: Explained Variance ──
print("\n[5] Explained variance ratio:")
total_var = eigvals.sum()
explained_ratio = eigvals / total_var
cumulative = np.cumsum(explained_ratio)
for i in range(min(10, len(eigvals))):
    print(f"    PC{i+1}: {explained_ratio[i]:.4f}  (cumulative: {cumulative[i]:.4f})")

# ── Step 6: Orthogonality Check ──
print("\n[6] Checking orthogonality of eigenvectors ...")
# Dot product of PC1 and PC2 should be ~0
dot_12 = np.dot(eigvecs[:, 0], eigvecs[:, 1])
dot_13 = np.dot(eigvecs[:, 0], eigvecs[:, 2])
dot_23 = np.dot(eigvecs[:, 1], eigvecs[:, 2])
print(f"    PC1 · PC2  = {dot_12:.6e}  (should be ~0)")
print(f"    PC1 · PC3  = {dot_13:.6e}  (should be ~0)")
print(f"    PC2 · PC3  = {dot_23:.6e}  (should be ~0)")

# Verify: Q^T Q should be Identity (for first 5 PCs)
Q = eigvecs[:, :5]
identity_check = Q.T @ Q
print(f"\n    Q^T Q (first 5 PCs, should be Identity):")
print(f"    {np.round(identity_check, 4)}")

# ── Step 7: Verify with sklearn ──
print("\n[7] Verification with sklearn PCA ...")
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(Z)
sklearn_ratio = pca.explained_variance_ratio_
print(f"    sklearn explained variance (first 5): {np.round(sklearn_ratio, 4)}")
print(f"    manual explained variance (first 5): {np.round(explained_ratio[:5], 4)}")
match = np.allclose(sklearn_ratio, explained_ratio[:5], atol=1e-4)
print(f"    Match: {match}")

print("\n" + "=" * 60)
print("PCA COMPLETE")
print("=" * 60)
