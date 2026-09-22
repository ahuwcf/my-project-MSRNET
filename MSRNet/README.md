 
```
MSRNet/
├── MSRNet.py                        # model definition (UM3F + CATE + MSR stages)
├── train.py                         # training entry point
├── test.py                          # standalone inference / evaluation (+ classification map)
├── datasets.py                      # dataset loading & HyperX patch dataset
├── utils_HSI.py                     # sampling, metrics, utilities
├── class_descriptions_houston.json  # Houston fine-grained class phrases (CATE text_1 input)
├── class_descriptions_hyrank.json   # HyRANK fine-grained class phrases
├── download_vit_b_32.py             # optional: save local CLIP ViT-B/32 state_dict
├── requirements.txt                 # pip environment
├── environment.yml                  # conda environment
├── preprocess/
│   ├── prepare_houston.py           # Houston13/18 raw data -> data/Houston/*.mat
│   └── prepare_hyrank.py            # Dioni/Loukia raw data -> data/HyRANK/*.mat
└── evaluation/
    ├── bench_msrnet.py              # Params / FLOPs / FPS benchmark
    ├── domain_metrics.py            # MMD & A-distance on raw spectra
    └── feature_domain_metrics.py    # MMD & A-distance on MSRNet features
```

## 1. Environment

**Option A: conda**

```bash
conda env create -f environment.yml
conda activate msrnet
```

**Option B: pip**

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Tested with Python 3.10, PyTorch 2.1 (CUDA 12.1), timm 0.4.12, on NVIDIA RTX 4090.

## 2. Data

Expected layout (default `--data_path ./data/`):

```
data/
├── Houston/
│   ├── Houston13.mat        # source domain HSI: ori_data (C, H, W), 48 bands
│   ├── Houston13_7gt.mat    # source ground truth: map (H, W), 7 classes, 0 = background
│   ├── Houston18.mat        # target domain HSI
│   └── Houston18_7gt.mat    # target ground truth
├── Pavia/                   # paviaU (source) / paviaC (target)
└── HyRANK/
    ├── Dioni.mat            # source domain HSI: ori_data (C, H, W), 176 bands
    ├── Dioni_7gt.mat        # source ground truth: map (H, W), 7 classes
    ├── Loukia.mat           # target domain HSI
    └── Loukia_7gt.mat       # target ground truth
```

### 2.1 From raw GRSS contest data

The Houston dataset is subject to the IEEE GRSS Data Fusion Contest license and is **not** redistributed here. Please obtain Houston 2013 / 2018 from GRSS, then run:

```bash
python preprocess/prepare_houston.py \
    --h13_img <raw Houston13 image .mat> --h13_gt <raw Houston13 gt .mat> \
    --h18_img <raw Houston18 image .mat> --h18_gt <raw Houston18 gt .mat>
```

The script maps the original 15 classes to the 7 classes used in this work and sub-samples the spectral bands to 48. **Please verify the `CLASS_MAP` and band-selection settings in the script against your acquisition before running** (documented inline).

### 2.2 HyRANK (Dioni -> Loukia)

HyRANK is distributed by the IEEE GRSS Data Fusion Technical Committee and is **not** redistributed here. After obtaining the raw Dioni / Loukia images and ground truths, run:

```bash
python preprocess/prepare_hyrank.py \
    --dioni_img <raw Dioni image .mat> --dioni_gt <raw Dioni gt .mat> \
    --loukia_img <raw Loukia image .mat> --loukia_gt <raw Loukia gt .mat>
```

Dioni serves as the source domain and Loukia as the target domain. **Please verify the 14-to-7 class mapping in the script against the class set used in the paper** (documented inline). Both images share the same 176-band sensor, so no band alignment is required.

### 2.3 Class text descriptions

`class_descriptions_houston.json` and `class_descriptions_hyrank.json` (the fine-grained phrases consumed by CATE) are included in the repository as static assets, so no generation step is needed for reproduction. Select the file matching the dataset via `--desc_file`.

## 3. Training

```bash
# Houston (source: Houston13, target: Houston18)
python train.py --dataset Houston --num_epoch 200 --batch_size 64 --lr 1e-3

# HyRANK (source: Dioni, target: Loukia)
python train.py --dataset HyRANK --desc_file class_descriptions_hyrank.json

# with the DIL domain-adversarial stage (mixes unlabeled target patches,
# domain labels: source=0, target=1)
python train.py --dataset Houston --domain_adversarial
```

Notes:

- Class text inputs are built from `class_descriptions_houston.json` (fine-grained phrases, `text_1`) and a global template *"a hyperspectral image of [CLASS]"* (`text_2`). At test time all class phrases are provided without label priors.
- The training log reports the modulation strengths λ₁ / λ₃ of the stage-aware modulator each epoch.
- Checkpoints are saved to `results/Houston_e<num_epoch>/best_model.pth` by best target-domain Kappa.
 
