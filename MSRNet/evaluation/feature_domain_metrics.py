# -*- coding: utf-8 -*-
"""MSRNet 特征空间域差异度量
- 加载训练 checkpoint 的 visual.* 权重 (默认 results/Houston_e200/best_model.pth,
  可通过环境变量 MSRNET_CKPT 指定)
- 按 train.py 相同预处理: img/img.max() + 逐像素 L2 归一化, 对称填充 r=8, patch 14x14
- 提取 512 维全局特征 -> MMD² (sigma=0.5) + A-distance (线性 SVM)
"""
import sys, os
import numpy as np
import torch
import h5py

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import clip
from clip.model import build_model

# 优先使用仓库内的 ViT-B-32.pt (state_dict 版本); 否则用官方 clip.load 自动下载
_ckpt = os.path.join(BASE, 'ViT-B-32.pt')
if os.path.exists(_ckpt):
    _sd_clip = torch.load(_ckpt, map_location='cpu')
    clip.load = lambda name='ViT-B/32', device='cuda', jit=False, download_root=None: (
        build_model(_sd_clip).to(device).eval().float(), None)

import MSRNet

BANDS, N_CLASSES, PATCH = 48, 7, 14
CKPT = os.environ.get('MSRNET_CKPT', os.path.join(BASE, 'results/Houston_e200/best_model.pth'))

model = MSRNet.MSRNet(embed_dim=512, bands=BANDS, num_classes=N_CLASSES,
                      num_blocks=4, clip_model_name='ViT-B/32').cuda()
model.eval()

# ---- 载入 visual 权重 ----
ckpt = torch.load(CKPT, map_location='cpu')
vis_sd = {k[len('visual.'):]: v for k, v in ckpt.items() if k.startswith('visual.')}
ref_sd = model.visual.state_dict()
missing = [k for k in ref_sd if k not in vis_sd]
extra = [k for k in vis_sd if k not in ref_sd]
shape_mismatch = [k for k in ref_sd if k in vis_sd and ref_sd[k].shape != vis_sd[k].shape]
print(f"visual 权重: ckpt {len(vis_sd)} 项 / 模型 {len(ref_sd)} 项, "
      f"missing={len(missing)}, extra={len(extra)}, shape 不符={len(shape_mismatch)}")
assert not missing and not extra and not shape_mismatch, "结构不匹配"
model.visual.load_state_dict(vis_sd)

# ---- 数据加载与预处理（与 get_dataset/train.py 一致）----
def load_and_preprocess(name):
    img = np.transpose(h5py.File(os.path.join(BASE, 'data', 'Houston', f'{name}.mat'), 'r')['ori_data']).astype('float32')
    gt = np.transpose(h5py.File(os.path.join(BASE, 'data', 'Houston', f'{name}_7gt.mat'), 'r')['map'])
    m, n, d = img.shape
    flat = img.reshape(m * n, -1)
    flat = flat / flat.max()                       # 全局 max 归一化（各数据集独立）
    l2 = np.sqrt((flat ** 2).sum(1, keepdims=True))
    l2[l2 == 0] = 1
    flat = flat / l2                               # 逐像素 L2 归一化
    img = flat.reshape(m, n, d)
    return img, np.asarray(gt)

def extract_patches(img, gt, coords):
    r = PATCH // 2 + 1
    img_p = np.pad(img, ((r, r), (r, r), (0, 0)), 'symmetric')
    patches = np.empty((len(coords), BANDS, PATCH, PATCH), dtype='float32')
    for i, (x, y) in enumerate(coords):
        x1, y1 = x - PATCH // 2 + r, y - PATCH // 2 + r
        patches[i] = img_p[x1:x1 + PATCH, y1:y1 + PATCH].transpose(2, 0, 1)
    return patches

def extract_features(patches, bs=256):
    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            x = torch.from_numpy(patches[i:i + bs]).cuda()
            g, _, _ = model.visual(x)
            feats.append(g.cpu().numpy())
    return np.concatenate(feats)

img_s, gt_s = load_and_preprocess('Houston13')
img_t, gt_t = load_and_preprocess('Houston18')
coords_s = np.argwhere(gt_s > 0)
coords_t = np.argwhere(gt_t > 0)

rng = np.random.RandomState(42)
n_t = min(5000, len(coords_t))
coords_t = coords_t[rng.choice(len(coords_t), n_t, replace=False)]
# 源域全部（2530 < 5000）

Fs = extract_features(extract_patches(img_s, gt_s, coords_s))
Ft = extract_features(extract_patches(img_t, gt_t, coords_t))
print(f"特征维度: {Fs.shape[1]}, 源域 {Fs.shape[0]} / 目标域 {Ft.shape[0]}")

# ================= MMD² (sigma=0.5) =================
def mmd2_rbf(X, Y, sigma):
    def sq(A, B):
        d = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.maximum(d, 0)
    m, n = len(X), len(Y)
    kxx = np.exp(-sq(X, X) / (2 * sigma**2))
    kyy = np.exp(-sq(Y, Y) / (2 * sigma**2))
    kxy = np.exp(-sq(X, Y) / (2 * sigma**2))
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    return (kxx.sum() / (m*(m-1)) + kyy.sum() / (n*(n-1)) - 2 * kxy.mean())

mmd2 = mmd2_rbf(Fs, Ft, sigma=0.5)

# ================= A-distance (线性 SVM) =================
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

Xd = np.vstack([Fs, Ft])
yd = np.r_[np.zeros(len(Fs)), np.ones(len(Ft))]
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.5, stratify=yd, random_state=42)
clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
clf.fit(Xtr, ytr)
err = 1 - clf.score(Xte, yte)
a_dist = 2 * (1 - 2 * err)

print()
print("├── MSRNet 特征上")
print(f"│   ├── MMD²: {mmd2:.4f}   (RBF, sigma=0.5)")
print(f"│   ├── A-distance: {a_dist:.4f}")
print(f"│   ├── 分类器测试误差 (ε): {err:.4f}")
print(f"│   └── 样本数量: 源域 {Fs.shape[0]} / 目标域 {Ft.shape[0]}")
