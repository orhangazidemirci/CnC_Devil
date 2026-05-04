"""
Functions for slicing data

NOTE: Going to refactor this with slice_train.py and spurious_train.py
      - Currently methods support different demos / explorations
"""
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from datasets import train_val_split, get_resampled_set, get_resampled_indices,initialize_data
from network import get_criterion, get_optim, get_net

import random

def random_exclude(n, exclude):
    candidates = [i for i in range(n + 1) if i != exclude]
    return random.choice(candidates)

def train_spurious_model(train_loaders, args, resample=False,
                         return_loaders=False, test_loader=None,
                         test_criterion=None):
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_loader, test_loader, visualize_dataset = initialize_data(args)
    del _
  
    net = get_net(args)
    optim = get_optim(net, args, model_type='spurious')
    criterion = get_criterion(args)
    
    log_test_results = True if test_loader is not None else False
    
    #SAVED_MODELS_DIR = f'./{args.dataset}/{args.arch}/saved_bias_models' # New dir for this version
    os.makedirs(args.bias_model_path, exist_ok=True)
        
    ##### N BIAS IMPLEMENTATION
        
    # bias_models = [ResNet18().to(DEVICE) for _ in range(num_bias_models)]
    bias_models=[]
    for i in range(args.num_bias_models):
        # MANUAL_SEED = 42
        # MANUAL_SEED = random.randint(0, 2**32 - 1)  # Generate a random 32-bit seed
    
        # random.seed(MANUAL_SEED)
        # np.random.seed(MANUAL_SEED)
        # torch.manual_seed(MANUAL_SEED)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(MANUAL_SEED)
        
        bias_models.append(get_net(args).to(DEVICE))

    
    
    # --- 7. Training and Validation Loop ---
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    for model_idx, model in enumerate(bias_models):
        # if model_idx>0:
        print(f"\n--- Training Bias Model {model_idx} ---")
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Using Adam 
        optim = get_optim(model, args, model_type='spurious')
        criterion = get_criterion(args)
        
        log_test_results = True if test_loader is not None else False
        
        
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        # For DEVIL, we train each bias model on a different slice of the data
        model_train_loader = train_loaders[model_idx]
        model_save_path = os.path.join(args.bias_model_path, f"bias_model_{model_idx}_best.pth")


        # best_val_accuracy_on_global_val = 0.0 # For saving the best version of this model
        best_val_loss_on_global_val = float('inf')
    
    
    ##############
        
        outputs = train_model(model, optim, criterion,
                            train_loader=model_train_loader,
                            val_loader=val_loader,
                            args=args, epochs=args.max_epoch_s,
                            log_test_results=log_test_results,
                            test_loader=test_loader,
                            test_criterion=test_criterion,
                            model_save_path=model_save_path,
                            model_idx=model_idx)
    if return_loaders:
        return bias_models, outputs, (train_loader_new, train_loader_spurious)
    return bias_models, outputs, None

