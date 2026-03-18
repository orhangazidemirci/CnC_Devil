"""
Datasets
"""
import copy
import numpy as np
import importlib
from torch.utils.data import DataLoader, Subset
import os
import pickle
def initialize_data(args):
    """
    Set dataset-specific arguments
    By default, the args.root_dir below should work ifinstalling datasets as
    specified in the README to the specified locations
    - Otherwise, change `args.root_dir` to the path where the data is stored.
    """
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')
    
    if 'waterbirds' in args.dataset:
        args.root_dir = '../slice-and-dice-smol/datasets/data/Waterbirds/'
        # args.root_dir = './datasets/data/Waterbirds/'
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ['landbirds', 'waterbirds']
        if args.dataset == 'waterbirds_r':
            args.train_classes = ['land', 'water']
            
    elif 'colored_mnist' in args.dataset:
        args.root_dir = './datasets/data/'
        args.data_path = './datasets/data/'
        args.target_name = 'digit'
        args.confounder_names = ['color']
        args.image_mean = 0.5
        args.image_std = 0.5
        args.augment_data = False
        # args.train_classes = args.train_classes
    
    elif 'celebA' in args.dataset:
        # args.root_dir = './datasets/data/CelebA/'  
        args.root_dir = '/dfs/scratch0/nims/CelebA/celeba/'
        # IMPORTANT - dataloader assumes that we have directory structure
        # in ./datasets/data/CelebA/ :
        # |-- list_attr_celeba.csv
        # |-- list_eval_partition.csv
        # |-- img_align_celeba/
        #     |-- image1.png
        #     |-- ...
        #     |-- imageN.png
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['blond', 'nonblond']
        args.val_split = 0.2
        
    elif 'civilcomments' in args.dataset:
        args.root_dir = './datasets/data/CivilComments/'
        args.target_name = 'toxic'
        args.confounder_names = ['identities']
        args.image_mean = 0
        args.image_std = 0
        args.augment_data = False
        args.image_path = './images/civilcomments/'
        args.train_classes = ['non_toxic', 'toxic']
        args.max_token_length = 300
        
    elif 'cxr' in args.dataset:
        args.root_dir = '/dfs/scratch1/ksaab/data/4tb_hdd/CXR'
        args.target_name = 'pmx'
        args.confounder_names = ['chest_tube']
        args.image_mean = 0.48865
        args.image_std = 0.24621
        args.augment_data = False
        args.image_path = './images/cxr/'
        args.train_classes = ['no_pmx', 'pmx']
    
    args.task = args.dataset  # e.g. 'civilcomments', for BERT
    args.num_classes = len(args.train_classes)
    return load_dataloaders, visualize_dataset


def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def get_resampled_indices(dataloader, args, sampling='subsample', seed=None):
    """
    Args:
    - dataloader (torch.utils.data.DataLoader): 
    - sampling (str): 'subsample' or 'upsample'
    """
    try:
        indices = dataloader.sampler.indices
    except:
        indices = np.arange(len(dataloader.dataset))
    indices = np.arange(len(dataloader.dataset))
    target_vals, target_val_counts = np.unique(
        dataloader.dataset.targets_all['target'][indices], 
        return_counts=True)
    sampled_indices = []
    if sampling == 'subsample':
        sample_size = np.min(target_val_counts)
    elif sampling == 'upsample':
        sample_size = np.max(target_val_counts)
    else:
        return indices
        
    if seed is None:
        seed = args.seed
    np.random.seed(seed)
    for v in target_vals:
        group_indices = np.where(
            dataloader.dataset.targets_all['target'][indices] == v)[0]
        if sampling == 'subsample':
            sampling_size = np.min([len(group_indices), sample_size])
            replace = False
        elif sampling == 'upsample':
            sampling_size = np.max([0, sample_size - len(group_indices)])
            sampled_indices.append(group_indices)
            replace = True
        sampled_indices.append(np.random.choice(
            group_indices, size=sampling_size, replace=replace))
    sampled_indices = np.concatenate(sampled_indices)
    np.random.seed(seed)
    np.random.shuffle(sampled_indices)
    return indices[sampled_indices]


def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain spurious dataset resampled_set
    Args:
    - dataset (torch.utils.data.Dataset): Spurious correlations dataset
    - resampled_set_indices (int[]): List-like of indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:  # Some dataset classes may not have these attributes
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.y_array
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except:
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
        
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            pass
    
    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]
        
    print('len(resampled_set.targets)', len(resampled_set.targets))
    return resampled_set


def initialize_data(args):
    """
    Set dataset-specific arguments.
    If args.devil=True, partitions training data into N subsets with varying bias ratios.
    If args.devil=False, returns classical single train/val/test loaders.
    """
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')
    args.root_dir = 'C:/Users/Orhan/Documents/GitHub/correct-n-contrast/datasets/data/Waterbirds/'
    if 'waterbirds' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = '../slice-and-dice-smol/datasets/data/Waterbirds/'
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ['landbirds', 'waterbirds']
        if args.dataset == 'waterbirds_r':
            args.train_classes = ['land', 'water']
        args.dataset_type = 'bias_benchmark'

    elif 'colored_mnist' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = './datasets/data/'
        args.data_path = './datasets/data/'
        args.target_name = 'digit'
        args.confounder_names = ['color']
        args.image_mean = 0.5
        args.image_std = 0.5
        args.augment_data = False
        args.dataset_type = 'bias_benchmark'

    elif 'celebA' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = '/dfs/scratch0/nims/CelebA/celeba/'
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['blond', 'nonblond']
        args.val_split = 0.2
        args.dataset_type = 'bias_benchmark'

    elif 'civilcomments' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = './datasets/data/CivilComments/'
        args.target_name = 'toxic'
        args.confounder_names = ['identities']
        args.image_mean = 0
        args.image_std = 0
        args.augment_data = False
        args.image_path = './images/civilcomments/'
        args.train_classes = ['non_toxic', 'toxic']
        args.max_token_length = 300
        args.dataset_type = 'bias_benchmark'

    elif 'cxr' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = '/dfs/scratch1/ksaab/data/4tb_hdd/CXR'
        args.target_name = 'pmx'
        args.confounder_names = ['chest_tube']
        args.image_mean = 0.48865
        args.image_std = 0.24621
        args.augment_data = False
        args.image_path = './images/cxr/'
        args.train_classes = ['no_pmx', 'pmx']
        args.dataset_type = 'bias_benchmark'

    else:
        args.dataset_type = 'standard'

    args.task = args.dataset
    args.num_classes = len(args.train_classes)

    if args.devil:
        return _initialize_devil(args, visualize_dataset)
    else:
        return load_dataloaders, visualize_dataset

def _initialize_devil(args, visualize_dataset):
    """
    Devil mode: load dataset via the standard load_dataloaders, then partition
    the training set into N subsets with varying spurious correlation strengths.

    Returns:
        train_loaders (list of DataLoader): One per bias model partition
        val_loader (DataLoader)
        test_loader (DataLoader)
        visualize_dataset (callable)
    """
    dataset_module = importlib.import_module(f'datasets.{args.dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')

    # Use the existing per-dataset loader to get train/val/test
    train_loader_full, val_loader, test_loader = load_dataloaders(args)

    indices_file = f'{args.dataset}_devil_split_{args.num_bias_models}_seed{args.seed}.pkl'

    if os.path.exists(indices_file):
        print(f"Loading cached devil partition indices from {indices_file}")
        with open(indices_file, 'rb') as f:
            all_train_indices = pickle.load(f)
    else:
        print(f"Computing devil stratified partitions for {args.dataset}...")
        all_train_indices = stratified_partition_for_bias_diversity(
            dataset=train_loader_full.dataset,
            num_bias_models=args.num_bias_models,
            dataset_type=args.dataset_type,
            seed=args.seed
        )
        with open(indices_file, 'wb') as f:
            pickle.dump(all_train_indices, f)
        print(f"Saved devil partition indices to {indices_file}")

    train_loaders = [
        DataLoader(
            Subset(train_loader_full.dataset, idx),
            batch_size=args.bs_trn,
            shuffle=True,
            num_workers=train_loader_full.num_workers
        )
        for idx in all_train_indices
    ]

    print(f"\nDevil mode ON — {args.dataset}, {args.num_bias_models} partitions:")
    for i, idx in enumerate(all_train_indices):
        print(f"  Partition {i}: {len(idx)} training samples")
    print(f"  Val samples:  {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    return train_loaders, val_loader, test_loader, visualize_dataset


def stratified_partition_for_bias_diversity(dataset, num_bias_models, dataset_type='standard', seed=42):
    np.random.seed(seed)
    num_total_samples = len(dataset)
    has_groups = hasattr(dataset, 'metadata_array') or hasattr(dataset, '_metadata_array')

    if dataset_type == 'bias_benchmark' and has_groups:
        print("Using STRATIFIED partitioning for bias benchmark dataset")
        return _stratified_partition_with_groups(dataset, num_bias_models, seed)
    else:
        print("Using RANDOM partitioning for standard dataset")
        return _random_partition(num_total_samples, num_bias_models, seed)


def _random_partition(num_total_samples, num_bias_models, seed):
    np.random.seed(seed)
    all_indices = list(range(num_total_samples))
    np.random.shuffle(all_indices)
    split_indices_np = np.array_split(np.array(all_indices), num_bias_models)
    return [list(arr) for arr in split_indices_np]


def _stratified_partition_with_groups(dataset, num_bias_models, seed):
    np.random.seed(seed)

    if hasattr(dataset, 'metadata_array'):
        metadata = dataset.metadata_array
    elif hasattr(dataset, '_metadata_array'):
        metadata = dataset._metadata_array
    else:
        raise ValueError("Dataset does not have metadata_array attribute")

    group_labels = metadata[:, 0].numpy() if hasattr(metadata, 'numpy') else np.array(metadata[:, 0])
    num_groups = len(np.unique(group_labels))
    group_counts = np.array([(group_labels == g).sum() for g in range(num_groups)])

    print(f"  Group counts: {group_counts.tolist()}")

    # Alternate groups are spurious/conflicting (0,2 spurious; 1,3 conflicting)
    # This matches Waterbirds and CelebA structure
    spurious_groups   = np.array([i for i in range(num_groups) if i % 2 == 0])  # 0, 2
    conflicting_groups = np.array([i for i in range(num_groups) if i % 2 == 1])  # 1, 3

    print(f"  Spurious groups:          {spurious_groups.tolist()}")
    print(f"  Bias-conflicting groups:  {conflicting_groups.tolist()}")

    # Separate indices by group
    group_indices = {}
    for group_idx in range(num_groups):
        indices = np.where(group_labels == group_idx)[0]
        np.random.shuffle(indices)
        group_indices[group_idx] = indices.tolist()

    bias_ratios = np.linspace(0.95, 0.50, num_bias_models)
    partitions = [[] for _ in range(num_bias_models)]

    # Use conflicting group size to bound partition size 
    # so we don't run out of minority samples
    total_conflicting = sum(len(group_indices[g]) for g in conflicting_groups)
    samples_per_partition = (total_conflicting * 2) // num_bias_models  # at most 50/50

    print(f"  Samples per partition (bounded): {samples_per_partition}")

    for partition_idx in range(num_bias_models):
        bias_ratio = bias_ratios[partition_idx]

        n_conflicting = int(samples_per_partition * (1 - bias_ratio))
        n_spurious    = samples_per_partition - n_conflicting

        n_spurious_per_group    = n_spurious    // len(spurious_groups)
        n_conflicting_per_group = n_conflicting // len(conflicting_groups)

        for group_idx in spurious_groups:
            n_take = min(n_spurious_per_group, len(group_indices[group_idx]))
            if n_take > 0:
                partitions[partition_idx].extend(group_indices[group_idx][:n_take])
                group_indices[group_idx] = group_indices[group_idx][n_take:]

        for group_idx in conflicting_groups:
            n_take = min(n_conflicting_per_group, len(group_indices[group_idx]))
            if n_take > 0:
                partitions[partition_idx].extend(group_indices[group_idx][:n_take])
                group_indices[group_idx] = group_indices[group_idx][n_take:]

        np.random.shuffle(partitions[partition_idx])
        print(f"  Partition {partition_idx}: {len(partitions[partition_idx])} samples, "
              f"target bias: {bias_ratio:.0%}")

    # Distribute remaining proportionally by target bias ratio
        remaining_spurious = []
        remaining_conflicting = []
        for group_idx in spurious_groups:
            remaining_spurious.extend(group_indices[group_idx])
        for group_idx in conflicting_groups:
            remaining_conflicting.extend(group_indices[group_idx])
        
        np.random.shuffle(remaining_spurious)
        np.random.shuffle(remaining_conflicting)
    
        # Split remaining spurious — more to high-bias partitions
        spurious_weights = bias_ratios / bias_ratios.sum()
        conflicting_weights = (1 - bias_ratios) / (1 - bias_ratios).sum()
    
        spurious_splits = np.round(spurious_weights * len(remaining_spurious)).astype(int)
        conflicting_splits = np.round(conflicting_weights * len(remaining_conflicting)).astype(int)
    
        # Fix rounding errors
        spurious_splits[-1] += len(remaining_spurious) - spurious_splits.sum()
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