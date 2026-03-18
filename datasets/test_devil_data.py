# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 05:58:12 2026

@author: Orhan
"""
import sys
sys.path.insert(0, 'C:/Users/Orhan/Documents/GitHub/correct-n-contrast')

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from datasets import initialize_data




def test_devil_mode():
    """Test devil vs classical mode initialization"""
    import argparse
    
    # Mock args
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    # Missing args that waterbirds.py expects
    args.arch = 'resnet50'
    args.bs_trn = 128
    args.bs_val = 128
    args.num_workers = 0
    args.image_path = './images/waterbirds/'
    args.root_dir = 'C:/Users/Orhan/Documents/GitHub/correct-n-contrast/datasets/data/Waterbirds/'
    args.val_split = 0.2
    args.log_path = './logs/'
    args.experiment_name = 'test'
    # Dataset args
    args.dataset = 'waterbirds'
    args.devil = True

    args.seed = 42
    args.batch_size = 128
    args.num_bias_models = 5
    args.no_cuda = True
    
    # ==========================================
    # TEST 1: Classical mode
    # ==========================================
    print("="*60)
    print("TEST 1: Classical mode (devil=False)")
    print("="*60)
    
    args.devil = False
    try:
        load_dataloaders, visualize_dataset = initialize_data(args)
        loaders = load_dataloaders(args, train_shuffle=False)
        train_loader, val_loader, test_loader = loaders

        dataset = train_loader.dataset
        print(type(dataset))
        print([attr for attr in dir(dataset) if not attr.startswith('__')])
        
        # Check loaders are single DataLoaders
        assert isinstance(train_loader, DataLoader), "train_loader should be a DataLoader"
        assert isinstance(val_loader, DataLoader),   "val_loader should be a DataLoader"
        assert isinstance(test_loader, DataLoader),  "test_loader should be a DataLoader"
        
        # Check a batch loads correctly
        x, y, meta = next(iter(train_loader))
        print(f"  ✓ Batch shape:  {x.shape}")
        print(f"  ✓ Labels shape: {y.shape}")
        print(f"  ✓ Meta shape:   {meta.shape}")
        print(f"  ✓ Train samples: {len(train_loader.dataset)}")
        print(f"  ✓ Val samples:   {len(val_loader.dataset)}")
        print(f"  ✓ Test samples:  {len(test_loader.dataset)}")
        print("  ✓ Classical mode PASSED\n")
        
    except Exception as e:
        print(f"  ✗ Classical mode FAILED: {e}\n")
        raise
    
    # ==========================================
    # TEST 2: Devil mode
    # ==========================================
    print("="*60)
    print("TEST 2: Devil mode (devil=True)")
    print("="*60)
    
    args.devil = True
    try:
        train_loaders, val_loader, test_loader, visualize_dataset = initialize_data(args)
        
        # Check we got a list of N loaders
        assert isinstance(train_loaders, list),                    "train_loaders should be a list"
        assert len(train_loaders) == args.num_bias_models,         f"Expected {args.num_bias_models} loaders, got {len(train_loaders)}"
        assert all(isinstance(l, DataLoader) for l in train_loaders), "All items should be DataLoaders"
        
        # Check val/test are still single loaders
        assert isinstance(val_loader, DataLoader),  "val_loader should be a DataLoader"
        assert isinstance(test_loader, DataLoader), "test_loader should be a DataLoader"
        
        # Check each partition loads a batch correctly
        total_samples = 0
        for i, loader in enumerate(train_loaders):
            x, y, meta = next(iter(loader))
            total_samples += len(loader.dataset)
            print(f"  Partition {i}: {len(loader.dataset)} samples | batch shape: {x.shape}")
        
        print(f"\n  ✓ Total partitioned samples: {total_samples}")
        print(f"  ✓ Val samples:               {len(val_loader.dataset)}")
        print(f"  ✓ Test samples:              {len(test_loader.dataset)}")
        
        # Check bias ratios are actually different across partitions
        print(f"\n  Checking bias diversity across partitions...")
        bias_ratios = []
        
        for i, loader in enumerate(train_loaders):
            subset = loader.dataset          # Subset
            dataset = subset.dataset         # Waterbirds directly
            indices = subset.indices
            groups = dataset.metadata_array[indices, 0].numpy()
            spurious = ((groups == 0) | (groups == 2)).sum()
            ratio = spurious / len(indices)
            bias_ratios.append(ratio)
            print(f"    Partition {i}: bias ratio = {ratio:.3f}")
        
        
        assert len(set([round(r, 2) for r in bias_ratios])) > 1, "Bias ratios should differ across partitions!"
        print(f"\n  ✓ Bias ratios are diverse: {[round(r,3) for r in bias_ratios]}")
        print("  ✓ Devil mode PASSED\n")
        
    except Exception as e:
        print(f"  ✗ Devil mode FAILED: {e}\n")
        raise


if __name__ == '__main__':
    
    test_devil_mode()
    