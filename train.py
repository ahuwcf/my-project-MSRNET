from __future__ import print_function
import argparse, json, torch, math, numpy as np, os, time
import torch.optim as optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import HyperX, get_dataset
from model_sfd_mgn import EHSnet

parser = argparse.ArgumentParser(description='SFD-MGN Training')
parser.add_argument('--save_path', type=str, default="./results/")
parser.add_argument('--data_path', type=str, default='./datasets/')
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
args = parser.parse_args()
DEVICE = get_device(args.cuda)

datasets_list = {
    'Houston': {'source_name': 'Houston13', 'target_name': 'Houston18',
                'source_label_name': 'Houston13_7gt', 'target_label_name': 'Houston18_7gt'},
}

def train(epoch, model, optimizer):
    model.train()
    correct = 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader
    for i in range(1, num_iter):
        data_src, label_src = iter_source.__next__()
        data_src = data_src.to(DEVICE).float()
        label_src = label_src.to(DEVICE).long() - 1
        optimizer.zero_grad()
        loss, logits = model(data_src, None, label_src, label_src)
        loss.backward()
        optimizer.step()
        pred = logits.data.max(1)[1]
        correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
        if i % args.log_interval == 0:
            print(f'Epoch {epoch} [{i*len(data_src)}/{len_src_dataset} ({100.*i/len_src_loader:.0f}%)] loss: {loss.item():.4f}')
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
            _, logits = model(data, None, label, label)
            pred = logits.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    pred_all = np.concatenate(pred_list)
    label_all = np.concatenate(label_list)
    results = metrics(pred_all, label_all, ignored_labels=hyperparams['ignored_labels'], n_classes=int(gt_src.max()))
    kappa = results['Kappa']
    aa = results['AA']
    acc = 100.*correct/len_tar_dataset
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
    img_src, gt_src, _, ignored_labels = get_dataset(source_name, args.data_path)
    print('load target dataset:', end=' ')
    img_tar, gt_tar, _, _ = get_dataset(target_name, args.data_path)
    num_classes = int(gt_src.max())
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels, 'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = {k:v for k,v in hyperparams.items() if v is not None}
    r = int(hyperparams['patch_size']/2)+1
    img_src = np.pad(img_src, ((r,r),(r,r),(0,0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r,r),(r,r),(0,0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r,r),(r,r)), 'constant', constant_values=0)
    gt_tar = np.pad(gt_tar, ((r,r),(r,r)), 'constant', constant_values=0)
    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    for _ in range(args.re_ratio-1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], pin_memory=True, worker_init_fn=seed_worker, generator=g, shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset, pin_memory=True, batch_size=64)
    len_src_loader = len(train_loader); len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    print(hyperparams)
    print(f'train: {len_src_dataset}, test: {len_tar_dataset}')
    with open(args.desc_file, 'r', encoding='utf-8') as f:
        class_descriptions = json.load(f)
    print(f'Loaded LLM descriptions for {len(class_descriptions)} classes')
    model = EHSnet(embed_dim=512, bands=N_BANDS, num_classes=num_classes, num_blocks=4, clip_model_name='ViT-B/32').to(DEVICE)
    model.set_class_descriptions(class_descriptions)
    print('Enhanced text prototypes computed')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params/(1024*1024):.2f}M params')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_kappa = 0
    for epoch in range(1, args.num_epoch+1):
        t1 = time.time()
        model = train(epoch, model, optimizer)
        t2 = time.time()
        kappa, aa = test(model)
        t3 = time.time()
        print(f'Train: {t2-t1:.1f}s, Test: {t3-t2:.1f}s')
        if kappa > best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
        print(f'Epoch {epoch}: Kappa={kappa:.4f}, Best={best_kappa:.4f}')
    print(f'Best Kappa: {best_kappa:.4f}')