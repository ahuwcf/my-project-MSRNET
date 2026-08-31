 
"""
import argparse
import numpy as np
import scipy.io as sio

# ---------- HyRANK 原始 14 类 -> 7 类映射 (按需修改!) ----------
# 原始 14 类:
#  1 Dense urban fabric, 2 Mineral extraction sites, 3 Non-irrigated arable land,
#  4 Vineyards, 5 Broad-leaved forest, 6 Coniferous forest, 7 Mixed forest,
#  8 Sclerophyllous vegetation, 9 Sparsely vegetated areas, 10 Beaches-dunes-sands,
#  11 Coastal water, 12 Intertidal flat, 13 Water bodies, 14 Roads
# 本仓库保留的 7 类:
#  1 mineral extraction sites, 2 non-irrigated arable land, 3 broad-leaved forest,
#  4 coniferous forest, 5 mixed forest, 6 sclerophyllous vegetation,
#  7 sparsely vegetated areas
CLASS_MAP = {
    2: 1,   # mineral extraction sites
    3: 2,   # non-irrigated arable land
    5: 3,   # broad-leaved forest
    6: 4,   # coniferous forest
    7: 5,   # mixed forest
    8: 6,   # sclerophyllous vegetation
    9: 7,   # sparsely vegetated areas
    # 其余类别映射为背景 0, 不参与实验
}


def remap_gt(gt_raw):
    gt = np.zeros_like(gt_raw, dtype=np.int16)
    for src, dst in CLASS_MAP.items():
        gt[gt_raw == src] = dst
    return gt


def prepare(img_raw, gt_raw, out_img, out_gt):
    """img_raw: (H, W, C) 或 (C, H, W); gt_raw: (H, W)"""
    if img_raw.shape[0] < img_raw.shape[-1]:   # (C, H, W) -> (H, W, C)
        img_raw = np.transpose(img_raw, (1, 2, 0))
    img = img_raw.astype(np.float32)
    gt = remap_gt(gt_raw)
    sio.savemat(out_img, {'ori_data': np.transpose(img, (2, 0, 1))}, do_compression=True)
    sio.savemat(out_gt, {'map': gt}, do_compression=True)
    print(f'{out_img}: bands={img.shape[-1]}, shape={img.shape[:2]}, '
          f'classes={sorted(np.unique(gt).tolist())}, samples={int((gt > 0).sum())}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dioni_img', type=str, required=True, help='原始 Dioni 图像 .mat')
    parser.add_argument('--dioni_gt', type=str, required=True, help='原始 Dioni 地真 .mat')
    parser.add_argument('--loukia_img', type=str, required=True, help='原始 Loukia 图像 .mat')
    parser.add_argument('--loukia_gt', type=str, required=True, help='原始 Loukia 地真 .mat')
    parser.add_argument('--img_key', type=str, default=None, help='图像 .mat 中的变量名 (默认自动探测)')
    parser.add_argument('--gt_key', type=str, default=None, help='地真 .mat 中的变量名 (默认自动探测)')
    parser.add_argument('--out_dir', type=str, default='./data/HyRANK')
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

    prepare(load_mat(args.dioni_img, args.img_key), load_mat(args.dioni_gt, args.gt_key),
            f'{args.out_dir}/Dioni.mat', f'{args.out_dir}/Dioni_7gt.mat')
    prepare(load_mat(args.loukia_img, args.img_key), load_mat(args.loukia_gt, args.gt_key),
            f'{args.out_dir}/Loukia.mat', f'{args.out_dir}/Loukia_7gt.mat')

    print('\n预处理完成。请核对输出文件的类别分布与论文设置一致。')
