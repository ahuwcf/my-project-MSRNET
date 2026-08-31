from __future__ import print_function
"""MSRNet 独立推理/测试脚本
加载训练好的 checkpoint, 在目标域上评估 OA / Kappa / AA, 可选输出分类图。
"""
import argparse, json, torch, numpy as np, os
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import HyperX, get_dataset
from MSRNet import MSRNet

parser = argparse.ArgumentParser(description='MSRNet Inference / Evaluation')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='Houston')
parser.add_argument('--checkpoint', type=str, default='./results/Houston_e200/best_model.pth')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--patch_size', type=int, default=14)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=3667)
parser.add_argument('--desc_file', type=str, default='class_descriptions_houston.json')
parser.add_argument('--save_pred', type=str, default='',
                    help='若非空, 将分类预测图保存为该路径的 .npy 文件')
args = parser.parse_args()
DEVICE = get_device(args.cuda)

datasets_list = {
    'Houston': {'source_name': 'Houston13', 'target_name': 'Houston18'},
    'Pavia': {'source_name': 'paviaU', 'target_name': 'paviaC'},
    'HyRANK': {'source_name': 'Dioni', 'target_name': 'Loukia'},
}
TEXT_TEMPLATE = 'a hyperspectral image of {}'
NEUTRAL_DESC = 'a hyperspectral image of land cover'

if __name__ == '__main__':
    seed_worker(args.seed)
    args.data_path = args.data_path + args.dataset + '/'
    source_name = datasets_list[args.dataset]['source_name']
    target_name = datasets_list[args.dataset]['target_name']

    # 源域仅用于确定类别数/波段数与类别名
    img_src, gt_src, label_values, ignored_labels = get_dataset(source_name, args.data_path)
    img_tar, gt_tar, _, _ = get_dataset(target_name, args.data_path)
    num_classes = int(gt_src.max())
    N_BANDS = img_src.shape[-1]

    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full',
                        'flip_augmentation': False, 'radiation_augmentation': False,
                        'mixture_augmentation': False})
    hyperparams = {k: v for k, v in hyperparams.items() if v is not None}
    r = int(hyperparams['patch_size'] / 2) + 1
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=0)
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset, batch_size=hyperparams['batch_size'], pin_memory=True)

    # 类别文本
    with open(args.desc_file, 'r', encoding='utf-8') as f:
        class_descriptions = json.load(f)
    assert len(class_descriptions) == num_classes
    all_phrases = sorted({p for phrases in class_descriptions for p in phrases})
    print(f'{num_classes} classes, {len(all_phrases)} unique phrases')

    # 模型与 checkpoint
    model = MSRNet(embed_dim=512, bands=N_BANDS, num_classes=num_classes,
                   num_blocks=4, clip_model_name='ViT-B/32').to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    try:
        model.load_state_dict(ckpt)
    except RuntimeError as e:
        raise RuntimeError(
            f'checkpoint 与模型结构不匹配: {e}\n'
            f'注意: 旧版本 checkpoint 不含 StageAwareModulator 等新模块权重, '
            f'需用修订后的代码重新训练; 或用 load_state_dict(ckpt, strict=False) 仅作参考。')
    model.eval()
    print(f'loaded checkpoint: {args.checkpoint}')

    # 推理
    correct = 0
    pred_list, label_list, coord_list = [], [], []
    with torch.no_grad():
        for inputs in test_loader:
            patch, label = inputs[0], inputs[1]
            patch = patch.to(DEVICE).float()
            label = label.to(DEVICE).long() - 1
            text_1 = [all_phrases] * patch.shape[0]   # 不使用类别先验
            text_2 = [NEUTRAL_DESC] * patch.shape[0]
            _, logits, _ = model(patch, text_1, text_2, label)
            pred = logits.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    pred_all = np.concatenate(pred_list)
    label_all = np.concatenate(label_list)

    results = metrics(pred_all, label_all, ignored_labels=ignored_labels, n_classes=num_classes)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'OA={acc:.2f}%, Kappa={results["Kappa"]:.4f}, AA={results["AA"]:.4f}')

    # 每类精度
    cm = results['Confusion_matrix']
    print('\nPer-class accuracy:')
    for c in range(num_classes):
        total_c = cm[c].sum()
        if total_c > 0:
            print(f'  {label_values[c]:>28s}: {100.*cm[c, c]/total_c:6.2f}%  (support {int(total_c)})')

    # 保存分类图 (可选)
    if args.save_pred:
        pred_map = np.zeros_like(gt_tar, dtype=np.int64)
        idx = np.argwhere(test_gt_tar > 0)
        for (x, y), p in zip(idx, pred_all):
            pred_map[x, y] = p + 1
        np.save(args.save_pred, pred_map)
        print(f'prediction map saved: {args.save_pred}')
