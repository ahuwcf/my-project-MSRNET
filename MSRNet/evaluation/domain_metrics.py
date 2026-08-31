# -*- coding: utf-8 -*-
"""Houston13 (源域) vs Houston18 (目标域) 的域差异度量：MMD 与 A-distance
- MMD: RBF 核（median 启发式带宽）+ 多带宽混合核两种口径，分块计算
- A-distance: A = 2(1 - 2ε)，ε 为域分类器的测试误差
样本：各自带标签的像素（gt > 0），与训练时用到的样本一致
"""
import os
import numpy as np
import h5py
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Houston')

def load(name):
    img = np.transpose(h5py.File(f'{BASE}/{name}.mat', 'r')['ori_data'])   # (H, W, C)
    gt = np.transpose(h5py.File(f'{BASE}/{name}_7gt.mat', 'r')['map'])
    mask = gt > 0
    return img[mask].astype(np.float64)

Xs_all = load('Houston13')
Xt_all = load('Houston18')
print(f"源域 Houston13: {Xs_all.shape[0]} 个带标签像素, {Xs_all.shape[1]} 波段")
print(f"目标域 Houston18: {Xt_all.shape[0]} 个带标签像素, {Xt_all.shape[1]} 波段")

rng = np.random.RandomState(42)
NS = min(5000, len(Xs_all)); NT = min(5000, len(Xt_all))
Xs_s = Xs_all[rng.choice(len(Xs_all), NS, replace=False)]
Xt_s = Xt_all[rng.choice(len(Xt_all), NT, replace=False)]

scaler = StandardScaler().fit(np.vstack([Xs_s, Xt_s]))
Zs = scaler.transform(Xs_s)
Zt = scaler.transform(Xt_s)
assert np.isfinite(Zs).all() and np.isfinite(Zt).all()

# ================= 1. MMD =================
def sq_dists(A, B):
    # (m, n) 欧氏距离平方，避免三维广播爆内存
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0)

def mmd2_rbf(X, Y, bandwidth):
    m, n = len(X), len(Y)
    d_xx = sq_dists(X, X); np.fill_diagonal(d_xx, 0)
    d_yy = sq_dists(Y, Y); np.fill_diagonal(d_yy, 0)
    d_xy = sq_dists(X, Y)
    kxx = np.exp(-d_xx / (2 * bandwidth**2))
    kyy = np.exp(-d_yy / (2 * bandwidth**2))
    kxy = np.exp(-d_xy / (2 * bandwidth**2))
    return ((kxx.sum() - m) / (m*(m-1))
            + (kyy.sum() - n) / (n*(n-1))
            - 2 * kxy.mean())

# median 启发式带宽（子采样 1000 估算）
Z = np.vstack([Zs[:1000], Zt[:1000]])
sub = Z[rng.choice(len(Z), 1000, replace=False)]
D = np.sqrt(sq_dists(sub, sub))
med = np.median(D[np.triu_indices(len(sub), 1)])
print(f"\nmedian 启发式带宽: {med:.4f}")

mmd2_med = mmd2_rbf(Zs, Zt, med)
print(f"MMD² (RBF, median 带宽):     {mmd2_med:.4f}")

mmd2_mk = np.mean([mmd2_rbf(Zs, Zt, s * med) for s in [0.25, 0.5, 1, 2, 4]])
print(f"MMD² (multi-kernel 5 带宽):  {mmd2_mk:.4f}")

# ================= 2. A-distance =================
def a_distance(make_clf, use_pca=None, n_train=1000):
    """用较少训练样本训练域分类器，避免 ε=0 导致 A-distance 饱和"""
    Xa, Xb = Zs, Zt
    if use_pca is not None:
        p = PCA(n_components=use_pca, random_state=0).fit(np.vstack([Xa, Xb]))
        Xa, Xb = p.transform(Xa), p.transform(Xb)
    Xd = np.vstack([Xa, Xb])
    yd = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.5, stratify=yd, random_state=42)
    clf = make_clf()
    clf.fit(Xtr[:n_train], ytr[:n_train])
    err = 1 - clf.score(Xte, yte)
    return 2 * (1 - 2 * err), err

print()
for n_train in [100, 500, 1000]:
    a, e = a_distance(lambda: LinearSVC(C=1.0, max_iter=5000, random_state=0), n_train=n_train)
    print(f"A-distance (线性SVM, 训练样本={n_train:>4}): {a:.4f}  (ε={e:.4f})")
a, e = a_distance(lambda: LogisticRegression(max_iter=2000, random_state=0), n_train=1000)
print(f"A-distance (逻辑回归, 训练样本=1000): {a:.4f}  (ε={e:.4f})")
a, e = a_distance(lambda: LinearSVC(C=1.0, max_iter=5000, random_state=0), use_pca=10, n_train=1000)
print(f"A-distance (线性SVM+PCA10, 训练样本=1000): {a:.4f}  (ε={e:.4f})")
# 全量训练（饱和上限参考）
a, e = a_distance(lambda: LinearSVC(C=1.0, max_iter=5000, random_state=0), n_train=5000)
print(f"A-distance (线性SVM, 全量 5000): {a:.4f}  (ε={e:.4f})")
