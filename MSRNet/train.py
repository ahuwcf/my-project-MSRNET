from __future__ import print_function
import argparse, json, torch, numpy as np, os, time
import torch.optim as optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import HyperX, get_dataset
from MSRNet import MSRNet

parser = argparse.ArgumentParser(description='MSRNet Training (Multi-Stage Semantic Reasoning Network)')
parser.add_argument('--save_path', type=str, default="./results/")
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='Houston')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--training_sample_ratio', type=float, default=0.8)
parser.add_argument('--re_ratio', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=14)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=3667)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--flip_augmentation', action='store_true', default=False)
parser.add_argument('--radiation_augmentation', action='store_true', default=False)
parser.add_argument('--mixture_augmentation', action='store_true', default=False)
parser.add_argument('--desc_file', type=str, default='class_descriptions_houston.json')
parser.add_argument('--domain_adversarial', action='store_true', default=False,
                    help='mix unlabeled target patches into each training batch and enable '
                         'the DIL adversarial domain loss (Stage 3)')
args = parser.parse_args()
DEVICE = get_device(args.cuda)

datasets_list = {
    'Houston': {'source_name': 'Houston13', 'target_name': 'Houston18',
                'source_label_name': 'Houston13_7gt', 'target_label_name': 'Houston18_7gt'},
    'Pavia': {'source_name': 'paviaU', 'target_name': 'paviaC',
              'source_label_name': 'paviaU_7gt', 'target_label_name': 'paviaC_7gt'},
    'HyRANK': {'source_name': 'Dioni', 'target_name': 'Loukia',
               'source_label_name': 'Dioni_7gt', 'target_label_name': 'Loukia_7gt'},
}

# 全局描述模板 (CATE, Fig. 3)
TEXT_TEMPLATE = 'a hyperspectral image of {}'
NEUTRAL_DESC = 'a hyperspectral image of land cover'

# 由数据集与描述文件在 main 中填充
CLASS_PHRASES = None      # CLASS_PHRASES[c]: 类别 c 的细粒度短语列表 (text_1)
CLASS_TEMPLATES = None    # CLASS_TEMPLATES[c]: 类别 c 的全局描述 (text_2)
ALL_PHRASES = None        # 测试时使用的全部类别短语集合


def train(epoch, model, optimizer):
    model.train()
    correct = 0
    iter_source = iter(train_loader)
    num_iter = len(train_loader)
    iter_target = iter(train_tar_loader) if args.domain_adversarial else None
    for i in range(1, num_iter + 1):
        data_src, label_src = next(iter_source)
        data_src = data_src.to(DEVICE).float()
        label_src = label_src.to(DEVICE).long() - 1

        # 文本输入: 短语取自该样本类别的 patch-wise 标注描述
        text_1 = [CLASS_PHRASES[int(l)] for l in label_src]
        text_2 = [CLASS_TEMPLATES[int(l)] for l in label_src]
        domain_label = None

        # 可选: 混入无标注目标域 patch, 启用 DIL 域对抗 (Stage 3)
        if args.domain_adversarial:
            try:
                data_tar, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(train_tar_loader)
                data_tar, _ = next(iter_target)
            data_tar = data_tar.to(DEVICE).float()
            bs = data_src.shape[0]
            data_src = torch.cat([data_src, data_tar], dim=0)
            # 目标域样本无类别标签: -100 会被 F.cross_entropy 默认忽略
            label_src = torch.cat([label_src,
                                   torch.full((data_tar.shape[0],), -100, dtype=torch.long, device=DEVICE)])
            text_1 = text_1 + [[] for _ in range(data_tar.shape[0])]
            text_2 = text_2 + [NEUTRAL_DESC] * data_tar.shape[0]
            domain_label = torch.cat([torch.zeros(bs, dtype=torch.long, device=DEVICE),
                                      torch.ones(data_tar.shape[0], dtype=torch.long, device=DEVICE)])

        optimizer.zero_grad()
        loss, logits, loss_dict = model(data_src, text_1, text_2, label_src, domain_label=domain_label)
        loss.backward()
        optimizer.step()
        pred = logits.data.max(1)[1]
        valid = label_src >= 0   # 域对抗模式下目标域样本标签为 -100, 不计入训练精度
        correct += pred[valid].eq(label_src[valid].view_as(pred[valid])).cpu().sum()
        if i % args.log_interval == 0:
            lam1 = loss_dict.get('lambda_1', float('nan'))
            lam3 = loss_dict.get('lambda_3', float('nan'))
            print(f'Epoch {epoch} [{i*args.batch_size}/{len_src_dataset} '
                  f'({100.*i/num_iter:.0f}%)] loss: {loss.item():.4f} '
                  f'λ1: {lam1:.3f} λ3: {lam3:.3f}')
    acc = correct.item() / len_src_dataset
    print(f'Epoch {epoch}: Train Acc={acc:.4f}')
    return model


def test(model):
    model.eval()
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(DEVICE).float()
            label = label.to(DEVICE).long() - 1
            # 测试时不使用类别先验: text_1 为全部类别的短语集合, text_2 为类别无关描述
            text_1 = [ALL_PHRASES] * data.shape[0]
            text_2 = [NEUTRAL_DESC] * data.shape[0]
            _, logits, _ = model(data, text_1, text_2, label)
            pred = logits.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    pred_all = np.concatenate(pred_list)
    label_all = np.concatenate(label_list)
    results = metrics(pred_all, label_all, ignored_labels=hyperparams['ignored_labels'], n_classes=int(gt_src.max()))
    kappa = results['Kappa']
    aa = results['AA']
    acc = 100. * correct / len_tar_dataset
    print(f'Test Acc={acc:.2f}%, Kappa={kappa:.4f}, AA={aa:.4f}')
    return kappa, aa


if __name__ == '__main__':
    args.save_path = args.save_path + args.dataset + f'_e{args.num_epoch}'
    os.makedirs(args.save_path, exist_ok=True)
    args.data_path = args.data_path + args.dataset + '/'
    source_name = datasets_list[args.dataset]['source_name']
    target_name = datasets_list[args.dataset]['target_name']
    seed_worker(args.seed)
    print('load source dataset:', end=' ')
    img_src, gt_src, label_values, ignored_labels = get_dataset(source_name, args.data_path)
    print('load target dataset:', end=' ')
    img_tar, gt_tar, _, _ = get_dataset(target_name, args.data_path)
    num_classes = int(gt_src.max())
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = {k: v for k, v in hyperparams.items() if v is not None}
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=0)
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=0)
    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    for _ in range(args.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True,
                              'mixture_augmentation': False})
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'],
                                   pin_memory=True, worker_init_fn=seed_worker, generator=g, shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset, pin_memory=True, batch_size=64)
    len_src_loader = len(train_loader); len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    print(hyperparams)
    print(f'train: {len_src_dataset}, test: {len_tar_dataset}')

    # 可选: 无标注目标域训练 loader (域对抗)
    train_tar_loader = None
    if args.domain_adversarial:
        tar_gt_train, _, _, _ = sample_gt(gt_tar, 0.5, mode='random')
        train_tar_dataset = HyperX(img_tar, tar_gt_train, **hyperparams_train)
        train_tar_loader = data.DataLoader(train_tar_dataset, batch_size=hyperparams['batch_size'],
                                           pin_memory=True, shuffle=True, drop_last=True)

    # 类别文本描述
    with open(args.desc_file, 'r', encoding='utf-8') as f:
        class_descriptions = json.load(f)
    assert len(class_descriptions) == num_classes, \
        f'description file has {len(class_descriptions)} classes but dataset has {num_classes}'
    CLASS_PHRASES = [list(dict.fromkeys(phrases)) for phrases in class_descriptions]
    CLASS_TEMPLATES = [TEXT_TEMPLATE.format(name) for name in label_values]
    ALL_PHRASES = sorted({p for phrases in CLASS_PHRASES for p in phrases})
    print(f'Loaded class descriptions: {num_classes} classes, {len(ALL_PHRASES)} unique phrases')

    model = MSRNet(embed_dim=512, bands=N_BANDS, num_classes=num_classes,
                   num_blocks=4, clip_model_name='ViT-B/32').to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params / (1024 * 1024):.2f}M trainable params '
          f'({sum(p.numel() for p in model.parameters()) / (1024 * 1024):.2f}M total)')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_kappa = 0
    for epoch in range(1, args.num_epoch + 1):
        t1 = time.time()
        model = train(epoch, model, optimizer)
        t2 = time.time()
        kappa, aa = test(model)
        t3 = time.time()
        print(f'Train: {t2 - t1:.1f}s, Test: {t3 - t2:.1f}s')
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
        print(f'Epoch {epoch}: Kappa={kappa:.4f}, Best={best_kappa:.4f}')
    print(f'Best Kappa: {best_kappa:.4f}')
