 
"""
import argparse
import numpy as np
import scipy.io as sio

# ---------- 15 类 -> 7 类映射 (按需修改!) ----------
# 原始 Houston13 的 15 类标签:
#  1 Healthy grass, 2 Stressed grass, 3 Synthetic grass, 4 Trees, 5 Soil,
#  6 Water, 7 Residential, 8 Commercial, 9 Road, 10 Sharply angled road,
# 11 Highway, 12 Railway, 13 Parking lot 1, 14 Parking lot 2, 15 Tennis court
# 本仓库保留的 7 类:
#  1 grass healthy, 2 grass stressed, 3 trees, 4 water,
#  5 residential buildings, 6 non-residential buildings, 7 road
CLASS_MAP = {
    1: 1,   # healthy grass -> grass healthy
    2: 2,   # stressed grass -> grass stressed
    4: 3,   # trees -> trees
    6: 4,   # water -> water
    7: 5,   # residential -> residential buildings
    8: 6,   # commercial -> non-residential buildings
    9: 7,   # road -> road
    # 其余类别 (3,5,10-15) 映射为背景 0, 不参与实验
}

# ---------- 波段选择 (按需修改!) ----------
# 144 -> 48 波段: 每隔 3 个取 1 个 (步长 3)。若你使用了别的降采样方式
# (如 PCA 或区间平均), 请替换 BAND_INDICES 的计算。
BAND_STEP = 3


def remap_gt(gt_raw):
    gt = np.zeros_like(gt_raw, dtype=np.int16)
    for src, dst in CLASS_MAP.items():
        gt[gt_raw == src] = dst
    return gt


def prepare(img_raw, gt_raw, band_indices, out_img, out_gt):
    """img_raw: (H, W, C) 或 (C, H, W); gt_raw: (H, W)"""
    if img_raw.shape[0] < img_raw.shape[-1]:   # (C, H, W) -> (H, W, C)
        img_raw = np.transpose(img_raw, (1, 2, 0))
    img = img_raw[:, :, band_indices].astype(np.float32)
    gt = remap_gt(gt_raw)
    sio.savemat(out_img, {'ori_data': np.transpose(img, (2, 0, 1))}, do_compression=True)
    sio.savemat(out_gt, {'map': gt}, do_compression=True)
    print(f'{out_img}: bands={img.shape[-1]}, shape={img.shape[:2]}, '
          f'classes={sorted(np.unique(gt).tolist())}, samples={int((gt > 0).sum())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h13_img', type=str, required=True, help='原始 Houston13 图像 .mat (含 144 波段)')
    parser.add_argument('--h13_gt', type=str, required=True, help='原始 Houston13 地真 .mat')
    parser.add_argument('--h18_img', type=str, required=True, help='原始 Houston18 图像 .mat')
    parser.add_argument('--h18_gt', type=str, required=True, help='原始 Houston18 地真 .mat')
    parser.add_argument('--img_key', type=str, default=None, help='图像 .mat 中的变量名 (默认自动探测)')
    parser.add_argument('--gt_key', type=str, default=None, help='地真 .mat 中的变量名 (默认自动探测)')
    parser.add_argument('--out_dir', type=str, default='./data/Houston')
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    def load_mat(path, key):
        d = sio.loadmat(path)
        if key is not None:
            return d[key]
        for k, v in d.items():
            if not k.startswith('__') and hasattr(v, 'shape'):
                return v
        raise KeyError(f'cannot detect variable in {path}')

    # Houston13: 144 波段
    img13 = load_mat(args.h13_img, args.img_key)
    gt13 = load_mat(args.h13_gt, args.gt_key)
    bands13 = list(range(0, img13.shape[-1] if img13.shape[-1] < img13.shape[0] else img13.shape[0], BAND_STEP))
    prepare(img13, gt13, bands13,
            f'{args.out_dir}/Houston13.mat', f'{args.out_dir}/Houston13_7gt.mat')

    # Houston18: 50 波段, 与 Houston13 的 48 波段对齐
    # (Houston18 与 Houston13 波段范围不同, 需按中心波长插值/重采样到统一 48 波段;
    #  若你已有对齐好的 48 波段版本, 直接改名放入 data/Houston 即可)
    img18 = load_mat(args.h18_img, args.img_key)
    gt18 = load_mat(args.h18_gt, args.gt_key)
    n18 = img18.shape[-1] if img18.shape[-1] < img18.shape[0] else img18.shape[0]
    bands18 = np.linspace(0, n18 - 1, len(bands13)).astype(int).tolist()
    prepare(img18, gt18, bands18,
            f'{args.out_dir}/Houston18.mat', f'{args.out_dir}/Houston18_7gt.mat')

    print('\n预处理完成。请核对输出文件的波段数/类别分布与论文设置一致。')
