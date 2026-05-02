# Devil-NET

**Devil-NET** is a debiasing framework that leverages multiple intentionally biased models to mitigate spurious correlations — without requiring group annotations, contrastive batch construction, or multi-stage annotation pipelines.

Built on top of [Correct-N-Contrast (CnC)](https://github.com/HazyResearch/correct-n-contrast), Devil-NET replaces the single ERM spurious model with N intentionally over-biased models, each trained on a different shard of the data with a different directional bias. The key insight: **a model overfitted to spurious correlations in shard A is unbiased to spurious correlations in shard B**, when A and B are disjoint.

---

## Key Idea

Standard debiasing methods identify spurious features and suppress them. Devil-NET instead exploits them:

1. **Partition** training data into N disjoint shards, each with a different per-class bias ratio
2. **Train** N bias models — one per shard — each intentionally overfitting to its shard's spurious patterns
3. **For each shard**, the N−1 models that never saw it act as unbiased *Angel* teachers; the own model is the *Devil*
4. **Train** the final model via a contrastive objective: pull representations toward Angel embeddings, push away from Devil embeddings

```
Shard 0: Devil = M0 (biased toward class 0)   Angels = M1, M2, M3, M4
Shard 1: Devil = M1 (biased toward class 1)   Angels = M0, M2, M3, M4
...
```

---

## Comparison with CnC

| | CnC | Devil-NET |
|---|---|---|
| Spurious model | 1 ERM model | N intentionally biased models |
| Positive signal | Same class, diff ERM pred | Pull toward Angel embedding |
| Negative signal | Diff class, same ERM pred | Push away from Devil embedding |
| Batch construction | Explicit slice sampling | Inline pair classification |
| Group labels required | No | No |
| Stages | 2 | 2 |

---

## Loss Function

The total training objective is:

```
L_total = L_CE  +  α · L_self  +  β · L_batch
```

**Self loss (Table 1)** — per-sample, based on Angel/Devil correctness:

| Angel correct | Devil correct | Loss |
|---|---|---|
| ✓ | ✗ | `w2·angel_pos + w3·devil_neg` (strongest signal) |
| ✓ | ✓ | `w1·angel_pos` |
| ✗ | ✗ | `w4·devil_neg` |
| ✗ | ✓ | `w5·devil_neg` (conservative, w5 ≪ w4) |

Where `angel_pos = -sim(z, z_angel)` and `devil_neg = +sim(z, z_devil)`.

**Batch loss (Table 2)** — SupCon-style over all pairs in the batch:

```
L_batch = -log( Σ w_pos · exp(sim(z, z_angel′)/τ) / (Σ w_pos·exp_pos + Σ w_neg·exp_neg) )
```

Pair roles are determined by true label, Angel prediction, and Devil prediction:

| Same class | Same Angel pred | Same Devil pred | Role |
|---|---|---|---|
| ✓ | ✗ | ✗ | Hard positive (w_A) |
| ✓ | ✓ | ✗ | Soft positive (w_B) |
| ✓ | ✓ | ✓ | Easy positive (w_C) |
| ✗ | ✗ | ✓ | Hard negative (w_E) |
| ✗ | ✓ | ✓ | Soft negative (w_F) |

---

## Installation

```bash
git clone https://github.com/orhangazidemirci/CnC_Devil.git
cd CnC_Devil
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.10+, numpy, scipy, tqdm, umap-learn, scikit-learn

---

## Datasets

Supported datasets (same as CnC):

| Dataset | Task | Spurious feature |
|---|---|---|
| Waterbirds | Bird classification | Background (land/water) |
| CelebA | Hair color | Gender |
| CivilComments | Toxicity detection | Identity mentions |
| ColoredMNIST | Digit classification | Color |
| CXR | Pneumothorax detection | Chest tube presence |

Download datasets and place under `datasets/data/`. Set `--root_dir` if using a custom path.

---

## Usage

### Stage 1 — Train bias models

```bash
python train_spurious.py \
    --dataset waterbirds \
    --arch resnet50_pt \
    --num_bias_models 5 \
    --max_epoch_s 1 \
    --lr_s 1e-3 \
    --bs_trn_s 32
```

Trained models are saved to `model/{dataset}/{arch}/saved_bias_models/`.

### Stage 2 — Train Devil-NET

```bash
python train_supervised_contrast.py \
    --dataset waterbirds \
    --arch resnet50_pt \
    --devil \
    --num_bias_models 5 \
    --max_epoch 10 \
    --lr 1e-4 \
    --temperature 0.07 \
    --alpha 1.0 \
    --beta 1.0 \
    --hard_neg_factor 0.0 \
    --bs_trn 128 \
    --optim sgd \
    --weight_decay 1e-4 \
    --weight_decay_c 1e-2
```

### Evaluate

```bash
python train_supervised_contrast.py \
    --dataset waterbirds \
    --arch resnet50_pt \
    --evaluate \
    --load_encoder <checkpoint_name>
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--devil` | `True` | Enable Devil-NET mode |
| `--num_bias_models` | `5` | Number of shards / bias models |
| `--temperature` | `0.05` | Contrastive temperature τ |
| `--alpha` | `1.0` | Self loss weight |
| `--beta` | `1.0` | Batch contrastive loss weight |
| `--hard_neg_factor` | `0.0` | Dynamic hard negative reweighting (0 = off) |
| `--max_epoch_s` | `1` | Epochs to train each bias model |
| `--pretrained_spurious_path` | `''` | Load pretrained bias models (skip stage 1) |

---

## Data Partitioning

For bias benchmark datasets (Waterbirds, CelebA), Devil-NET uses **cross-class stratified partitioning**: each shard is assigned a different per-class bias ratio while maintaining the global average bias (default 0.95) across all shards.

```
Partition 0: class 0 bias = 1.00,  class 1 bias = 0.90  (avg = 0.95)
Partition 2: class 0 bias = 0.95,  class 1 bias = 0.95  (avg = 0.95)
Partition 4: class 0 bias = 0.90,  class 1 bias = 1.00  (avg = 0.95)
```

This ensures each Devil model learns a **different directional shortcut**, so Angel models have orthogonal blind spots.

Partition indices are cached to disk after first computation:
```
{dataset}_devil_split_{num_bias_models}_seed{seed}.pkl
```

---

## Project Structure

```
CnC_Devil/
├── train_supervised_contrast.py  # Main entry point (stage 2)
├── train_spurious.py             # Bias model training (stage 1)
├── devil_net_train.py            # Training + evaluation loop
├── devil_net_loss.py             # DevilNetLoss (self + batch contrastive)
├── initialize_data.py            # Data loading + partitioning
├── get_optim.py                  # Optimizer factory
├── save_activations.py           # Mid-layer feature extraction
├── datasets/
│   ├── waterbirds.py
│   ├── celebA.py
│   ├── civilcomments.py
│   ├── colored_mnist.py
│   └── cxr.py
└── models/
    └── ...
```

---
