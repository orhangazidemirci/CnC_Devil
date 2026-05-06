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
from network import get_net, get_optim, save_checkpoint
from activations import save_activations


def get_targets_all(dataset):
    """Unwrap Subset to get targets_all"""
    if hasattr(dataset, 'dataset'):  # Subset
        return dataset.dataset.targets_all
    return dataset.targets_all

    
def compute_devil_net_signals(bias_models, dataloader, data_idx, args):
    # Get targets aligned with what the dataloader actually sees
    dataset = dataloader.dataset
    if hasattr(dataset, 'indices'):
        full_targets = get_targets_all(dataset.dataset)['target']
        targets      = full_targets[dataset.indices]
    else:
        targets = get_targets_all(dataset)['target']

    # Devil
    devil_embeddings, devil_predictions = save_activations(
        bias_models[data_idx], dataloader, args)

    n_samples = len(devil_predictions)   # ground truth size from actual inference
    targets   = np.array(targets[:n_samples])

    # Angels
    angel_emb_list  = []
    angel_pred_list = []
    for model_idx, model in enumerate(bias_models):
        if model_idx == data_idx:
            continue
        emb, pred = save_activations(model, dataloader, args)
        angel_emb_list.append(emb)
        angel_pred_list.append(pred)

    if not angel_emb_list:
        raise ValueError(f"No Angel models for shard {data_idx}. Need >= 2 shards.")

    # Best Angel per sample
    emb_dim           = angel_emb_list[0].shape[1]
    angel_embeddings  = np.zeros((n_samples, emb_dim), dtype=np.float32)
    angel_predictions = np.zeros(n_samples, dtype=np.int64)

    for sample_idx in range(n_samples):
        best_emb  = None
        best_pred = None
        best_conf = -1.0

        for emb, pred in zip(angel_emb_list, angel_pred_list):
            if pred[sample_idx] == targets[sample_idx]:
                conf = np.linalg.norm(emb[sample_idx])
                if conf > best_conf:
                    best_conf = conf
                    best_emb  = emb[sample_idx]
                    best_pred = pred[sample_idx]

        if best_emb is not None:
            angel_embeddings[sample_idx]  = best_emb
            angel_predictions[sample_idx] = best_pred
        else:
            angel_embeddings[sample_idx]  = np.mean(
                [e[sample_idx] for e in angel_emb_list], axis=0)
            angel_predictions[sample_idx] = angel_pred_list[0][sample_idx]

    return {
        'angel_predictions': angel_predictions.astype(np.int64),
        'devil_predictions': devil_predictions.astype(np.int64),
        'angel_embeddings':  angel_embeddings.astype(np.float32),
        'devil_embeddings':  devil_embeddings.astype(np.float32),
        'targets':           targets.astype(np.int64),
    }



def run_tests(N_MODELS=4, N_SAMPLES=80, N_CLASSES=4, INPUT_DIM=16, EMBED_DIM=32):
    print(f"\n{'='*60}")
    print(f"  compute_devil_net_signals — test suite")
    print(f"  N_MODELS={N_MODELS}  N_SAMPLES={N_SAMPLES}  "
          f"N_CLASSES={N_CLASSES}  INPUT_DIM={INPUT_DIM}")
    print(f"{'='*60}\n")
    from types import SimpleNamespace
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

# ---- Test 4 & 5: Angel = best Angel per sample (highest norm among correct) ----
        angel_indices = [i for i in range(N_MODELS) if i != data_idx]
        expected_angel_pred = np.zeros(N_SAMPLES, dtype=np.int64)
        expected_angel_emb  = np.zeros((N_SAMPLES, EMBED_DIM), dtype=np.float32)

        for sample_idx in range(N_SAMPLES):
            best_emb  = None
            best_pred = None
            best_conf = -1.0

            for i in angel_indices:
                if raw[i]['pred'][sample_idx] == dataset.targets[sample_idx]:
                    conf = np.linalg.norm(raw[i]['emb'][sample_idx])
                    if conf > best_conf:
                        best_conf = conf
                        best_emb  = raw[i]['emb'][sample_idx]
                        best_pred = raw[i]['pred'][sample_idx]

            if best_emb is not None:
                expected_angel_emb[sample_idx]  = best_emb
                expected_angel_pred[sample_idx] = best_pred
            else:
                # fallback: mean embedding, first Angel's prediction
                expected_angel_emb[sample_idx]  = np.mean(
                    [raw[i]['emb'][sample_idx] for i in angel_indices], axis=0)
                expected_angel_pred[sample_idx] = raw[angel_indices[0]]['pred'][sample_idx]

        np.testing.assert_array_equal(
            out['angel_predictions'], expected_angel_pred,
            err_msg="FAIL: angel_predictions don't match best Angel")
        print(f"  [PASS] Angel predictions = best Angel over models {angel_indices}")
        passed += 1

        np.testing.assert_allclose(
            out['angel_embeddings'], expected_angel_emb, rtol=1e-5,
            err_msg="FAIL: angel_embeddings don't match best Angel")
        print(f"  [PASS] Angel embeddings = best Angel over models {angel_indices}")
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
    
    from types import SimpleNamespace
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader


    class SaveOutput:
        def __init__(self): self.outputs = []
        def __call__(self, module, input, output): self.outputs.append(output.detach().cpu())
        def clear(self): self.outputs = []


    class DummyDataset(Dataset):
        def __init__(self, n_samples=100, n_classes=4, input_dim=16):
            self.n_samples = n_samples
            self.data      = torch.randn(n_samples, input_dim)
            self.targets   = np.random.randint(0, n_classes, n_samples).astype(np.int64)
        def __len__(self): return self.n_samples
        def __getitem__(self, idx): return self.data[idx], int(self.targets[idx]), idx


    class DummyModel(nn.Module):
        def __init__(self, input_dim=16, embed_dim=32, n_classes=4, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.encoder    = nn.Linear(input_dim, embed_dim)
            self.classifier = nn.Linear(embed_dim, n_classes)
            self.activation_layer = 'encoder'
        def forward(self, x):
            return self.classifier(self.encoder(x))
        
    # Standard case
    def get_targets_all(dataset):
        """Unwrap Subset/ConcatDataset to get targets_all"""
        # Unwrap Subset
        if hasattr(dataset, 'dataset'):
            return get_targets_all(dataset.dataset)  # recurse in case of nested Subsets
        # ConcatDataset — merge from all sub-datasets
        if hasattr(dataset, 'datasets'):
            all_targets = np.concatenate([
                get_targets_all(d)['target'] for d in dataset.datasets])
            all_spurious = np.concatenate([
                get_targets_all(d)['spurious'] for d in dataset.datasets])
            return {'target': all_targets, 'spurious': all_spurious}
        # Has targets_all directly
        if hasattr(dataset, 'targets_all'):
            return dataset.targets_all
        # Fallback for DummyDataset or other test datasets
        if hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
            return {'target': targets, 'spurious': np.zeros_like(targets)}
        raise AttributeError(f"Cannot extract targets_all from {type(dataset)}")
        
    run_tests(N_MODELS=4)

    # Larger N
    run_tests(N_MODELS=6, N_SAMPLES=120)