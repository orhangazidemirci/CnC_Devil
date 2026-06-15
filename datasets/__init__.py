"""
Datasets
"""

import os
import copy
import pickle
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits.

    Args:
        dataset   : torch Dataset
        val_split : fraction allocated to validation
        seed      : reproducibility seed

    Returns:
        train_indices, val_indices : np.ndarray, np.ndarray
    """
    train_ix    = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    return all_indices[train_ix:], all_indices[:train_ix]


def get_resampled_indices(dataloader, args, sampling='subsample', seed=None):
    """
    Args:
        dataloader : torch DataLoader
        sampling   : 'subsample' or 'upsample'
    """
    # Fixed: removed redundant unconditional reassignment
    try:
        indices = dataloader.sampler.indices
    except AttributeError:
        indices = np.arange(len(dataloader.dataset))

    target_vals, target_val_counts = np.unique(
        dataloader.dataset.targets_all['target'][indices],
        return_counts=True)

    if sampling == 'subsample':
        sample_size = np.min(target_val_counts)
    elif sampling == 'upsample':
        sample_size = np.max(target_val_counts)
    else:
        return indices

    if seed is None:
        seed = args.seed
    np.random.seed(seed)

    sampled_indices = []
    for v in target_vals:
        group_indices = np.where(
            dataloader.dataset.targets_all['target'][indices] == v)[0]
        if sampling == 'subsample':
            sampling_size = min(len(group_indices), sample_size)
            replace = False
        else:  # upsample
            sampling_size = max(0, sample_size - len(group_indices))
            sampled_indices.append(group_indices)
            replace = True
        sampled_indices.append(
            np.random.choice(group_indices, size=sampling_size, replace=replace))

    sampled_indices = np.concatenate(sampled_indices)
    np.random.shuffle(sampled_indices)
    return indices[sampled_indices]


def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Return dataset reindexed to resampled_set_indices.
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:
        resampled_set.y_array     = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets     = resampled_set.y_array
        try:
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except AttributeError:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError:
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except Exception:
            resampled_set_indices = np.concatenate(resampled_set_indices)
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        try:
            resampled_set.df = resampled_set.df.iloc[resampled_set_indices]
        except AttributeError:
            pass
        try:
            resampled_set.data = resampled_set.data[resampled_set_indices]
        except AttributeError:
            pass
        try:
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except AttributeError:
            pass

    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]

    print(f'Resampled dataset size: {len(resampled_set.targets)}')
    return resampled_set


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def initialize_data(args):
    """
    Set dataset-specific args and return data loaders.

    Returns:
        train_loaders    : list of DataLoaders (N partitions if devil, else [single loader])
        val_loader       : DataLoader
        test_loader      : DataLoader
        visualize_dataset: callable
    """
    dataset_module    = importlib.import_module(f'datasets.{args.dataset}')
    load_dataloaders  = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')

    # --- Dataset-specific args ---
    if 'waterbirds' in args.dataset:
        # args.root_dir = 'C:/Users/Orhan/Documents/GitHub/CnC_Devil/datasets/data/Waterbirds/'


        
        args.root_dir = getattr(args, 'root_dir', None)
        if not args.root_dir or 'slice-and-dice' in args.root_dir:
            args.root_dir = './datasets/data/Waterbirds/'

        args.target_name     = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.image_mean      = np.mean([0.485, 0.456, 0.406])
        args.image_std       = np.mean([0.229, 0.224, 0.225])
        args.augment_data    = False
        args.train_classes   = ['land', 'water'] if args.dataset == 'waterbirds_r' \
                               else ['landbirds', 'waterbirds']
        args.dataset_type    = 'bias_benchmark'

    elif 'colored_mnist' in args.dataset:
        args.root_dir        = getattr(args, 'root_dir', None) or './datasets/data/'
        args.data_path       = './datasets/data/'
        args.target_name     = 'digit'
        args.confounder_names = ['color']
        args.image_mean      = 0.5
        args.image_std       = 0.5
        args.augment_data    = False
        args.dataset_type    = 'bias_benchmark'

    elif 'celebA' in args.dataset:
        args.root_dir        = getattr(args, 'root_dir', None) or \
                               '/dfs/scratch0/nims/CelebA/celeba/'
        args.target_name     = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean      = np.mean([0.485, 0.456, 0.406])
        args.image_std       = np.mean([0.229, 0.224, 0.225])
        args.augment_data    = False
        args.train_classes   = ['blond', 'nonblond']
        args.val_split       = 0.2
        args.dataset_type    = 'bias_benchmark'

    elif 'civilcomments' in args.dataset:
        args.root_dir        = getattr(args, 'root_dir', None) or \
                               './datasets/data/CivilComments/'
        args.target_name     = 'toxic'
        args.confounder_names = ['identities']
        args.image_mean      = 0
        args.image_std       = 0
        args.augment_data    = False
        args.train_classes   = ['non_toxic', 'toxic']
        args.max_token_length = 300
        args.dataset_type    = 'bias_benchmark'

    elif 'cxr' in args.dataset:
        args.root_dir        = getattr(args, 'root_dir', None) or \
                               '/dfs/scratch1/ksaab/data/4tb_hdd/CXR'
        args.target_name     = 'pmx'
        args.confounder_names = ['chest_tube']
        args.image_mean      = 0.48865
        args.image_std       = 0.24621
        args.augment_data    = False
        args.train_classes   = ['no_pmx', 'pmx']
        args.dataset_type    = 'bias_benchmark'

    else:
        args.dataset_type = 'standard'

    args.task = args.dataset

    # Fixed: guard against standard datasets where train_classes may not be set
    args.num_classes = len(args.train_classes) \
                       if hasattr(args, 'train_classes') and args.train_classes else 2

    if args.devil:
        return _initialize_devil(args, load_dataloaders, visualize_dataset)

    # Non-devil: wrap single loader in a list for consistent interface
    train_loader, val_loader, test_loader = load_dataloaders(args)
    return train_loader, val_loader, test_loader, visualize_dataset


# ---------------------------------------------------------------------------
# Devil mode initialization
# ---------------------------------------------------------------------------

def _initialize_devil(args, load_dataloaders, visualize_dataset):
    """
    Load dataset then partition training set into N shards with varying
    spurious correlation strengths.

    Returns:
        train_loaders    : list[DataLoader]  — one per partition, shuffle=False
        val_loader       : DataLoader
        test_loader      : DataLoader
        visualize_dataset: callable
    """
    train_loader_full, val_loader, test_loader = load_dataloaders(args)



    indices_file = (f'{args.dataset}_devil_split'
                    f'_{args.num_bias_models}_seed{args.seed}.pkl')


    if getattr(args, 'new_partition', False) and os.path.exists(indices_file):
        os.remove(indices_file)
        print(f'Deleted cached partition: {indices_file}')

    if os.path.exists(indices_file):
        print(f'Loading cached devil partition indices from {indices_file}')
        with open(indices_file, 'rb') as f:
            all_train_indices = pickle.load(f)
    else:
        print(f'Computing devil partitions for {args.dataset}...')
        all_train_indices = stratified_partition_for_bias_diversity(
            dataset=train_loader_full.dataset,
            num_bias_models=args.num_bias_models,
            dataset_type=args.dataset_type,
            seed=args.seed,
        )
        with open(indices_file, 'wb') as f:
            pickle.dump(all_train_indices, f)
        print(f'Saved devil partition indices to {indices_file}')

    # Fixed: shuffle=False — required so signal array indices stay aligned
    train_loaders = [
        DataLoader(
            Subset(train_loader_full.dataset, idx),
            batch_size=args.bs_trn,
            shuffle=False,
            num_workers=train_loader_full.num_workers,
        )
        for idx in all_train_indices
    ]

    print(f'\nDevil mode — {args.dataset}, {args.num_bias_models} partitions:')
    for i, idx in enumerate(all_train_indices):
        print(f'  Partition {i}: {len(idx)} samples')
    print(f'  Val:  {len(val_loader.dataset)} samples')
    print(f'  Test: {len(test_loader.dataset)} samples')

    return train_loaders, val_loader, test_loader, visualize_dataset


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def stratified_partition_for_bias_diversity(dataset, num_bias_models,
                                             dataset_type='standard',
                                             base_bias=0.95, seed=42):
    np.random.seed(seed)
    has_groups = (hasattr(dataset, 'metadata_array') or
                  hasattr(dataset, '_metadata_array'))

    if dataset_type == 'bias_benchmark' and has_groups:
        print('Using PER-CLASS BIAS partitioning (bias benchmark dataset)')
        return _stratified_partition_per_class_bias(
            dataset, num_bias_models, base_bias=base_bias, seed=seed)

    print('Using RANDOM partitioning (standard dataset)')
    return _random_partition(len(dataset), num_bias_models, seed)


def _random_partition(num_total_samples, num_bias_models, seed):
    np.random.seed(seed)
    all_indices = np.arange(num_total_samples)
    np.random.shuffle(all_indices)
    return [arr.tolist() for arr in np.array_split(all_indices, num_bias_models)]

def _stratified_partition_per_class_bias(dataset, num_bias_models,
                                          base_bias=0.95, seed=42):
    np.random.seed(seed)
    N = num_bias_models

    if hasattr(dataset, 'metadata_array'):
        metadata = dataset.metadata_array
    elif hasattr(dataset, '_metadata_array'):
        metadata = dataset._metadata_array
    else:
        raise ValueError('Dataset has no metadata_array attribute.')

    group_labels = (metadata[:, 0].numpy()
                    if hasattr(metadata, 'numpy')
                    else np.array(metadata[:, 0]))

    num_groups  = len(np.unique(group_labels))
    num_classes = num_groups // 2
    group_counts = np.array([(group_labels == g).sum() for g in range(num_groups)])
    print(f'  Group counts: {group_counts.tolist()}')

    group_indices = {}
    for g in range(num_groups):
        idx = np.where(group_labels == g)[0]
        np.random.shuffle(idx)
        group_indices[g] = idx.tolist()

    # Step 1 — identify spurious (majority) and conflicting (minority) per class
    class_spur_conf = {}
    for c in range(num_classes):
        g_a, g_b = 2 * c, 2 * c + 1
        if len(group_indices[g_a]) >= len(group_indices[g_b]):
            class_spur_conf[c] = (g_a, g_b)
        else:
            class_spur_conf[c] = (g_b, g_a)
        spur_g, conf_g = class_spur_conf[c]
        print(f'  Class {c}: spur=group{spur_g}(n={len(group_indices[spur_g])})  '
              f'conf=group{conf_g}(n={len(group_indices[conf_g])})')

    # Step 2 — compute per-class spread
    per_class_max_spread = []
    for c in range(num_classes):
        spur_g, conf_g = class_spur_conf[c]
        n_spur         = len(group_indices[spur_g])
        n_conf         = len(group_indices[conf_g])
        n_total_c      = n_spur + n_conf
        n_per_part_c   = n_total_c // N

        if n_conf < N:
            spread = 0.0
        else:
            spread_from_range = min(base_bias - 0.5, 1.0 - base_bias)
            spread_from_pool  = n_conf / n_per_part_c - (1.0 - base_bias)
            spread = max(min(spread_from_range, spread_from_pool), 0.0)

        per_class_max_spread.append(spread)
        print(f'  Class {c}: n_spur={n_spur} n_conf={n_conf} spread={spread:.4f}')

    # Step 3 — compute target biases per class per partition
    offsets_base = np.linspace(1.0, -1.0, N)
    class_biases = np.zeros((num_classes, N))
    for c in range(num_classes):
        direction       = 1 if c % 2 == 0 else -1
        class_biases[c] = np.clip(
            base_bias + direction * per_class_max_spread[c] * offsets_base,
            0.5, 1.0
        )

    print(f'\n  Target bias per class per partition:')
    for c in range(num_classes):
        vals = '  '.join(f'{v:.2f}' for v in class_biases[c])
        print(f'    Class {c}: [{vals}]  avg={class_biases[c].mean():.3f}')

    partitions = [[] for _ in range(N)]

    # Step 4 — allocate samples
    for c in range(num_classes):
        spur_g, conf_g = class_spur_conf[c]
        spur_pool = list(group_indices[spur_g])
        conf_pool = list(group_indices[conf_g])
        n_total    = len(spur_pool) + len(conf_pool)
        n_per_part = n_total // N

        # First pass: allocate n_per_part per partition at target bias
        for i in range(N):
            n_spur_i = min(int(round(class_biases[c][i] * n_per_part)), len(spur_pool))
            n_conf_i = min(n_per_part - n_spur_i, len(conf_pool))
            shortfall = (n_per_part - n_spur_i) - n_conf_i
            if shortfall > 0:
                n_spur_i = min(n_spur_i + shortfall, len(spur_pool))

            partitions[i].extend(spur_pool[:n_spur_i])
            partitions[i].extend(conf_pool[:n_conf_i])
            spur_pool = spur_pool[n_spur_i:]
            conf_pool = conf_pool[n_conf_i:]

        # Second pass: distribute remaining proportionally
        if spur_pool or conf_pool:
            spur_w = class_biases[c] / class_biases[c].sum()
            conf_w = (1 - class_biases[c])
            conf_w = conf_w / conf_w.sum() if conf_w.sum() > 0 else np.ones(N) / N

            spur_splits = np.round(spur_w * len(spur_pool)).astype(int)
            conf_splits = np.round(conf_w * len(conf_pool)).astype(int)
            spur_splits[-1] += len(spur_pool) - spur_splits.sum()
            conf_splits[-1] += len(conf_pool) - conf_splits.sum()

            s, k = 0, 0
            for i in range(N):
                partitions[i].extend(spur_pool[s: s + spur_splits[i]])
                partitions[i].extend(conf_pool[k: k + conf_splits[i]])
                s += spur_splits[i]
                k += conf_splits[i]

    # Shuffle and report
    print(f'\n  Actual bias per class per partition:')
    for i in range(N):
        np.random.shuffle(partitions[i])
        part_groups = group_labels[partitions[i]]
        row = []
        for c in range(num_classes):
            spur_g, conf_g = class_spur_conf[c]
            n_s    = (part_groups == spur_g).sum()
            n_c    = (part_groups == conf_g).sum()
            actual = n_s / (n_s + n_c) if (n_s + n_c) > 0 else 0.0
            row.append(f'c{c}={actual:.2f}(t={class_biases[c][i]:.2f})')
        print(f'    Partition {i} [{len(partitions[i])} samples]: {" | ".join(row)}')

    return partitions