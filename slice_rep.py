"""
Alternative ERM model predictions by clustering representations
"""
import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image
from itertools import permutations
from tqdm import tqdm

# Representation-based slicing
import umap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Use a scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Data
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from datasets import train_val_split, get_resampled_indices, get_resampled_set

# Logging and training
from slice import train_spurious_model
from utils.logging import log_data, initialize_csv_metrics
from utils import print_header

from utils.logging import summarize_acc, log_data
from utils.visualize import plot_confusion, plot_data_batch

# Model
from network import get_net, get_optim, get_criterion, save_checkpoint
from activations import save_activations


def get_targets_all(dataset):
    """Unwrap Subset to get targets_all"""
    if hasattr(dataset, 'dataset'):  # Subset
        return dataset.dataset.targets_all
    return dataset.targets_all

    

def compute_devil_net_signals(bias_models, dataloader, data_idx, args):
    """
    Returns per-sample Angel/Devil predictions and embeddings
    for use in the Devil-NET contrastive loss.
    """
    targets = get_targets_all(dataloader.dataset)['target']
    n_samples = len(targets)

    # --- Devil: own model (seen this shard) ---
    devil_embeddings, devil_predictions = save_activations(
        bias_models[data_idx], dataloader, args)

    # --- Angel: aggregate over all unseen models ---
    angel_embeddings_all = []
    angel_predictions_all = []

    for model_idx, model in enumerate(bias_models):
        if model_idx == data_idx:
            continue                          # skip Devil
        emb, pred = save_activations(model, dataloader, args)
        angel_embeddings_all.append(emb)
        angel_predictions_all.append(pred)

    # majority vote for Angel prediction
    angel_predictions_stack = np.stack(angel_predictions_all, axis=1)  # (N, n_angels)
    angel_predictions = mode(angel_predictions_stack, axis=1).mode.squeeze()

    # average embedding across all Angels
    angel_embeddings = np.mean(np.stack(angel_embeddings_all, axis=0), axis=0)  # (N, D)

    return {
        'angel_predictions': angel_predictions,   # â per sample
        'devil_predictions': devil_predictions,   # d̂ per sample
        'angel_embeddings':  angel_embeddings,    # z_angel per sample
        'devil_embeddings':  devil_embeddings,    # z_devil per sample
        'targets':           targets,
    }



def run_tests(N_MODELS=4, N_SAMPLES=80, N_CLASSES=4, INPUT_DIM=16, EMBED_DIM=32):
    print(f"\n{'='*60}")
    print(f"  compute_devil_net_signals — test suite")
    print(f"  N_MODELS={N_MODELS}  N_SAMPLES={N_SAMPLES}  "
          f"N_CLASSES={N_CLASSES}  INPUT_DIM={INPUT_DIM}")
    print(f"{'='*60}\n")

    args = SimpleNamespace(device=torch.device('cpu'))

    # Build shared dataset and dataloader
    dataset    = DummyDataset(N_SAMPLES, N_CLASSES, INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Build N dummy models with different seeds
    models = [DummyModel(INPUT_DIM, EMBED_DIM, N_CLASSES, seed=i) for i in range(N_MODELS)]

    # Collect raw embeddings/predictions from every model (ground truth for checks)
    raw = {}
    for i, m in enumerate(models):
        emb, pred = save_activations(m, dataloader, args)
        raw[i] = {'emb': emb, 'pred': pred}
        print(f"  Model {i} accuracy: "
              f"{(pred == dataset.targets).mean()*100:.1f}%  "
              f"embedding shape: {emb.shape}")

    print()

    passed = 0
    failed = 0

    for data_idx in range(N_MODELS):
        print(f"--- Partition {data_idx} (Devil = model {data_idx}) ---")

        out = compute_devil_net_signals(
            bias_models=models,
            dataloader=dataloader,
            data_idx=data_idx,
            args=args
        )

        # ---- Test 1: output keys ----
        expected_keys = {'angel_predictions','devil_predictions',
                         'angel_embeddings','devil_embeddings','targets'}
        assert set(out.keys()) == expected_keys, "FAIL: missing keys"
        print(f"  [PASS] output keys correct")
        passed += 1

        # ---- Test 2: shapes ----
        assert out['angel_predictions'].shape == (N_SAMPLES,),  "FAIL: angel_pred shape"
        assert out['devil_predictions'].shape == (N_SAMPLES,),  "FAIL: devil_pred shape"
        assert out['angel_embeddings'].shape  == (N_SAMPLES, EMBED_DIM), "FAIL: angel_emb shape"
        assert out['devil_embeddings'].shape  == (N_SAMPLES, EMBED_DIM), "FAIL: devil_emb shape"
        assert out['targets'].shape           == (N_SAMPLES,),  "FAIL: targets shape"
        print(f"  [PASS] all shapes correct  "
              f"(angel_emb={out['angel_embeddings'].shape}, "
              f"devil_emb={out['devil_embeddings'].shape})")
        passed += 1

        # ---- Test 3: Devil is excluded from Angel pool ----
        # Devil predictions must equal raw[data_idx] predictions exactly
        np.testing.assert_array_equal(
            out['devil_predictions'], raw[data_idx]['pred'],
            err_msg="FAIL: devil_predictions don't match own model")
        print(f"  [PASS] Devil predictions match model {data_idx} exactly")
        passed += 1

        # ---- Test 4: Angel prediction = majority vote over all OTHER models ----
        angel_indices  = [i for i in range(N_MODELS) if i != data_idx]
        pred_stack     = np.stack([raw[i]['pred'] for i in angel_indices], axis=1)
        expected_angel = mode(pred_stack, axis=1).mode.squeeze().astype(np.int64)
        np.testing.assert_array_equal(
            out['angel_predictions'], expected_angel,
            err_msg="FAIL: angel_predictions don't match majority vote")
        print(f"  [PASS] Angel predictions = majority vote over models "
              f"{angel_indices}")
        passed += 1

        # ---- Test 5: Angel embedding = mean over all OTHER models ----
        emb_stack     = np.stack([raw[i]['emb'] for i in angel_indices], axis=0)
        expected_emb  = emb_stack.mean(axis=0).astype(np.float32)
        np.testing.assert_allclose(
            out['angel_embeddings'], expected_emb, rtol=1e-5,
            err_msg="FAIL: angel_embeddings don't match mean")
        print(f"  [PASS] Angel embeddings = mean over models {angel_indices}")
        passed += 1

        # ---- Test 6: targets unchanged ----
        np.testing.assert_array_equal(
            out['targets'], dataset.targets,
            err_msg="FAIL: targets corrupted")
        print(f"  [PASS] targets unchanged")
        passed += 1

        print()

    # ---- Test 7: edge case — N=2 (only 1 Angel per partition) ----
    print("--- Edge case: N=2 models (1 Angel per partition) ---")
    models_2 = [DummyModel(INPUT_DIM, EMBED_DIM, N_CLASSES, seed=i) for i in range(2)]
    for did in range(2):
        out2 = compute_devil_net_signals(models_2, dataloader, did, args)
        assert out2['angel_embeddings'].shape == (N_SAMPLES, EMBED_DIM)
        print(f"  [PASS] partition {did}: N=2 works fine")
        passed += 1

    # ---- Test 8: N=1 should raise ----
    print("\n--- Edge case: N=1 model (no Angels possible) ---")
    models_1 = [DummyModel(INPUT_DIM, EMBED_DIM, N_CLASSES, seed=0)]
    try:
        compute_devil_net_signals(models_1, dataloader, 0, args)
        print("  [FAIL] should have raised ValueError")
        failed += 1
    except ValueError as e:
        print(f"  [PASS] correctly raised ValueError: {e}")
        passed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Standard case
    run_tests(N_MODELS=4)

    # Larger N
    run_tests(N_MODELS=6, N_SAMPLES=120)