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
        args.root_dir        = getattr(args, 'root_dir', None) or \
                               '../slice-and-dice-smol/datasets/data/Waterbirds/'
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
    return [train_loader], val_loader, test_loader, visualize_dataset


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
                                             dataset_type='standard', seed=42):
    np.random.seed(seed)
    has_groups = (hasattr(dataset, 'metadata_array') or
                  hasattr(dataset, '_metadata_array'))

    if dataset_type == 'bias_benchmark' and has_groups:
        print('Using STRATIFIED partitioning (bias benchmark dataset)')
        return _stratified_partition_with_groups(dataset, num_bias_models, seed)

    print('Using RANDOM partitioning (standard dataset)')
    return _random_partition(len(dataset), num_bias_models, seed)


def _random_partition(num_total_samples, num_bias_models, seed):
    np.random.seed(seed)
    all_indices = np.arange(num_total_samples)
    np.random.shuffle(all_indices)
    return [arr.tolist() for arr in np.array_split(all_indices, num_bias_models)]


def _stratified_partition_with_groups(dataset, num_bias_models, seed):
    """
    Partition dataset into N shards with linearly decreasing spurious bias ratios
    (from 95% down to 50%), so each bias model trains on a differently biased view.
    """
    np.random.seed(seed)

    if hasattr(dataset, 'metadata_array'):
        metadata = dataset.metadata_array
    elif hasattr(dataset, '_metadata_array'):
        metadata = dataset._metadata_array
    else:
        raise ValueError('Dataset has no metadata_array attribute.')

    group_labels = (metadata[:, 0].numpy()
                    if hasattr(metadata, 'numpy')
                    else np.array(metadata[:, 0]))
    num_groups   = len(np.unique(group_labels))
    group_counts = np.array([(group_labels == g).sum() for g in range(num_groups)])

    print(f'  Group counts: {group_counts.tolist()}')

    # Even group indices = spurious (majority), odd = conflicting (minority)
    # Matches Waterbirds (0:landbird/land, 1:landbird/water, 2:waterbird/water, 3:waterbird/land)
    # and CelebA (0:nonblond/female, 1:nonblond/male, 2:blond/female, 3:blond/male)
    spurious_groups    = np.array([i for i in range(num_groups) if i % 2 == 0])
    conflicting_groups = np.array([i for i in range(num_groups) if i % 2 == 1])

    print(f'  Spurious groups:    {spurious_groups.tolist()}')
    print(f'  Conflicting groups: {conflicting_groups.tolist()}')

    # Shuffle indices within each group
    group_indices = {}
    for g in range(num_groups):
        idx = np.where(group_labels == g)[0]
        np.random.shuffle(idx)
        group_indices[g] = idx.tolist()

    # Bias ratios: partition 0 most biased (0.95 spurious), last least biased (0.50)
    bias_ratios = np.linspace(0.95, 0.50, num_bias_models)

    # Bound partition size by minority pool so we don't exhaust conflicting samples
    total_conflicting    = sum(len(group_indices[g]) for g in conflicting_groups)
    samples_per_partition = (total_conflicting * 2) // num_bias_models
    print(f'  Samples per partition (bounded): {samples_per_partition}')

    partitions = [[] for _ in range(num_bias_models)]

    # --- First pass: fill each partition to target bias ratio ---
    for partition_idx in range(num_bias_models):
        ratio      = bias_ratios[partition_idx]
        n_spur     = int(samples_per_partition * ratio)
        n_conf     = samples_per_partition - n_spur
        n_spur_pg  = n_spur // len(spurious_groups)
        n_conf_pg  = n_conf // len(conflicting_groups)

        for g in spurious_groups:
            n_take = min(n_spur_pg, len(group_indices[g]))
            partitions[partition_idx].extend(group_indices[g][:n_take])
            group_indices[g] = group_indices[g][n_take:]

        for g in conflicting_groups:
            n_take = min(n_conf_pg, len(group_indices[g]))
            partitions[partition_idx].extend(group_indices[g][:n_take])
            group_indices[g] = group_indices[g][n_take:]

        np.random.shuffle(partitions[partition_idx])
        print(f'  Partition {partition_idx}: {len(partitions[partition_idx])} samples, '
              f'target bias: {ratio:.0%}')

    # --- Second pass: distribute remaining samples proportionally ---
    # Fixed: dedented out of the for loop — runs once after all partitions filled
    remaining_spurious    = []
    remaining_conflicting = []
    for g in spurious_groups:
        remaining_spurious.extend(group_indices[g])
    for g in conflicting_groups:
        remaining_conflicting.extend(group_indices[g])

    np.random.shuffle(remaining_spurious)
    np.random.shuffle(remaining_conflicting)

    # High-bias partitions get more spurious; low-bias get more conflicting
    spurious_weights    = bias_ratios / bias_ratios.sum()
    conflicting_weights = (1 - bias_ratios) / (1 - bias_ratios).sum()

    spurious_splits    = np.round(spurious_weights * len(remaining_spurious)).astype(int)
    conflicting_splits = np.round(conflicting_weights * len(remaining_conflicting)).astype(int)

    # Absorb rounding error into last partition
    spurious_splits[-1]    += len(remaining_spurious)    - spurious_splits.sum()
    conflicting_splits[-1] += len(remaining_conflicting) - conflicting_splits.sum()

    s_idx, c_idx = 0, 0
    for partition_idx in range(num_bias_models):
        s_take = spurious_splits[partition_idx]
        c_take = conflicting_splits[partition_idx]
        partitions[partition_idx].extend(remaining_spurious[s_idx:s_idx + s_take])
        partitions[partition_idx].extend(remaining_conflicting[c_idx:c_idx + c_take])
        s_idx += s_take
        c_idx += c_take
        np.random.shuffle(partitions[partition_idx])

    return partitions