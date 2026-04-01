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
from train import train_model, test_model
from network import get_criterion, get_optim, get_net, get_output


def compute_slice_indices(net, dataloader, criterion, 
                          batch_size, args, resample_by='class',
                          loss_factor=1., use_dataloader=False):
    """
    Use trained model to predict "slices" of data belonging to different subgroups

    Args:
    - net (torch.nn.Module): Pytorch neural network model
    - dataloader (torch.nn.utils.DataLoader): Pytorch data loader
    - criterion (torch.nn.Loss): Pytorch cross-entropy loss (with reduction='none')
    - batch_size (int): Batch size to compute slices over
    - args (argparse): Experiment arguments
    - resamble_by (str): How to resample, ['class', 'correct']
    Returns:
    - sliced_data_indices (int(np.array)[]): List of numpy arrays denoting indices of the dataloader.dataset
                                             corresponding to different slices
    """
    # First compute pseudolabels
    dataloader_ = dataloader if use_dataloader else None
    dataset = dataloader.dataset
    slice_outputs = compute_pseudolabels(net, dataset, 
                                         batch_size, args,  # Added this dataloader
                                         criterion, dataloader=dataloader_)
    pseudo_labels, outputs, correct, correct_spurious, losses = slice_outputs
    
    output_probabilities = torch.exp(outputs) / torch.exp(outputs).sum(dim=1).unsqueeze(dim=1)

    sliced_data_indices = []
    all_losses = []
    all_correct = []
    correct = correct.detach().cpu().numpy()
    all_probs = []
    for label in np.unique(pseudo_labels):
        group = np.where(pseudo_labels == label)[0]
        if args.weigh_slice_samples_by_loss:
            losses_per_group = losses[group]
        correct_by_group = correct[group]
        probs_by_group = output_probabilities[group]
        if args.subsample_labels is True or args.supersample_labels is True:
            group_vals = np.unique(dataloader.dataset.targets[group],
                                   return_counts=True)[1]
            sample_size = (np.min(group_vals) if args.subsample_labels is True
                           else np.max(group_vals))
            sampled_indices = []
            # These end up being the same
            if resample_by == 'class':
                target_values = dataloader.dataset.targets[group]
            elif resample_by == 'correct':
                target_values = correct_by_group
            # assert correct_by_group == dataloader.dataset.targets[group]
            print(f'> Resampling by {resample_by}...')
            for v in np.unique(target_values):
                group_indices = np.where(target_values == v)[0]
                if args.subsample_labels is True:
                    sampling_size = np.min([len(group_indices), sample_size])
                    replace = False
                    p = None
                elif args.supersample_labels is True:
                    sampling_size = np.max(
                        [0, sample_size - len(group_indices)])
                    sampled_indices.append(group_indices)
                    replace = True
                    if args.weigh_slice_samples_by_loss:
                        p = losses_per_group[group_indices] * loss_factor
                        p = (torch.exp(p) / torch.exp(p).sum()).numpy()
                    else:
                        p = None
                sampled_indices.append(np.random.choice(
                    group_indices, size=sampling_size, replace=replace, p=p)) 
            sampled_indices = np.concatenate(sampled_indices)
            sorted_indices = np.arange(len(sampled_indices))
            if args.weigh_slice_samples_by_loss:
                all_losses.append(losses_per_group[sampled_indices][sorted_indices])
            sorted_indices = np.arange(len(sampled_indices))
            sliced_data_indices.append(group[sampled_indices][sorted_indices])
            all_correct.append(correct_by_group[sampled_indices][sorted_indices])
            all_probs.append(probs_by_group[sampled_indices][sorted_indices])
        else:
            if args.weigh_slice_samples_by_loss:
                sorted_indices = torch.argsort(losses_per_group, descending=True)
                all_losses.append(losses_per_group[sorted_indices])
            else:
                sorted_indices = np.arange(len(group))
            sliced_data_indices.append(group[sorted_indices])
            all_correct.append(correct_by_group[sorted_indices])
            all_probs.append(probs_by_group[sorted_indices])
    # Save GPU memory
    for p in net.parameters():
        p = p.detach().cpu() 
    net.to(torch.device('cpu')) 
    return sliced_data_indices, all_losses, all_correct, all_probs


def compute_pseudolabels(net, dataset, batch_size, args, criterion=None, 
                         dataloader=None):
    net.eval()
    if dataloader is None:
        new_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers)
    else:
        new_loader = dataloader
        dataset = dataloader.dataset
    all_outputs = []
    all_predicted = []
    all_correct = []
    all_correct_spurious = []
    all_losses = []
    net.to(args.device)

    with torch.no_grad():
        targets_s = dataset.targets_all['spurious']
        for batch_ix, data in enumerate(tqdm(new_loader)):
            inputs, labels, data_ix = data
            labels_spurious = torch.tensor(
                [targets_s[ix] for ix in data_ix]).to(args.device)

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = get_output(net, inputs, labels, args)
            _, predicted = torch.max(outputs.data, 1)
            all_outputs.append(outputs.detach().cpu())
            all_predicted.append(predicted.detach().cpu())
            if args.weigh_slice_samples_by_loss:
                assert criterion is not None, 'Need to specify criterion'
                loss = criterion(outputs, labels)
                all_losses.append(loss.detach().cpu())

            # Save correct
            correct = (predicted == labels).to(torch.device('cpu'))
            correct_spurious = (predicted == labels_spurious).to(torch.device('cpu'))
            all_correct.append(correct)
            all_correct_spurious.append(correct_spurious)
            
            inputs = inputs.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))
            outputs = outputs.to(torch.device('cpu'))
            predicted = predicted.to(torch.device('cpu'))

    pseudo_labels = torch.hstack(all_predicted)
    outputs = torch.vstack(all_outputs)
    correct = torch.hstack(all_correct)
    correct_spurious = torch.hstack(all_correct_spurious)
    if len(all_losses) > 0:
        all_losses = torch.hstack(all_losses)
    else:
        all_losses = None
    return pseudo_labels, outputs, correct, correct_spurious, all_losses


def train_spurious_model(train_loaders, args, resample=False,
                         return_loaders=False, test_loader=None,
                         test_criterion=None):
    train_indices, train_indices_spurious = train_val_split(train_loaders.dataset,
                                                            val_split=args.spurious_train_split, 
                                                            seed=args.seed)
    
    train_targets_all = train_loaders.dataset.targets_all
    unique_target_counts = np.unique(train_targets_all['target'][train_indices_spurious],
                                     return_counts=True)
    print(f'Target values in spurious training data: {unique_target_counts}')
    
    train_set_new = get_resampled_set(train_loaders.dataset,
                                      train_indices,
                                      copy_dataset=True)
    train_set_spurious = get_resampled_set(train_loaders.dataset,
                                           train_indices_spurious,
                                           copy_dataset=True)

    train_loader_new = DataLoader(train_set_new,
                                  batch_size=args.bs_trn,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    train_loader_spurious = DataLoader(train_set_spurious,
                                       batch_size=args.bs_trn,
                                       shuffle=False,
                                       num_workers=args.num_workers)
    if resample is True:
        resampled_indices = get_resampled_indices(train_loader_spurious,
                                                  args,
                                                  args.resample_class)
        train_set_resampled = get_resampled_set(train_set_spurious,
                                                resampled_indices)
        train_loader_spurious = DataLoader(train_set_resampled,
                                           batch_size=args.bs_trn,
                                           shuffle=True,
                                           num_workers=args.num_workers)
        
    net = get_net(args)
    optim = get_optim(net, args, model_type='spurious')
    criterion = get_criterion(args, reduction='mean')

    
    log_test_results = True if test_loader is not None else False

    outputs = train_model(net, optim, criterion,
                          train_loader=train_loader_spurious,
                          val_loader=train_loader_new,
                          args=args, epochs=args.max_epoch_s,
                          log_test_results=log_test_results,
                          test_loader=test_loader,
                          test_criterion=test_criterion)
    
    if return_loaders:
        return net, outputs, (train_loader_new, train_loader_spurious)
    return net, outputs, None

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
    if args.devil:
        pass
    else:
        # Classical path — unchanged
        train_indices, train_indices_spurious = train_val_split(train_loaders.dataset,
                                                                val_split=args.spurious_train_split,
                                                                seed=args.seed)
        train_targets_all = train_loaders.dataset.targets_all
        unique_target_counts = np.unique(train_targets_all['target'][train_indices_spurious],
                                         return_counts=True)
        print(f'Target values in spurious training data: {unique_target_counts}')
    
        train_set_new = get_resampled_set(train_loaders.dataset,
                                          train_indices,
                                          copy_dataset=True)
        train_set_spurious = get_resampled_set(train_loaders.dataset,
                                               train_indices_spurious,
                                               copy_dataset=True)
        train_loader_new = DataLoader(train_set_new,
                                      batch_size=args.bs_trn,
                                      shuffle=False,
                                      num_workers=args.num_workers)
        train_loader_spurious = DataLoader(train_set_spurious,
                                           batch_size=args.bs_trn,
                                           shuffle=False,
                                           num_workers=args.num_workers)
        if resample is True:
            resampled_indices = get_resampled_indices(train_loader_spurious,
                                                      args,
                                                      args.resample_class)
            train_set_resampled = get_resampled_set(train_set_spurious,
                                                    resampled_indices)
            train_loader_spurious = DataLoader(train_set_resampled,
                                               batch_size=args.bs_trn,
                                               shuffle=True,
                                               num_workers=args.num_workers)
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
        
        # if args.arch=="resnet":
        #     bias_models.append(get_net(args).to(DEVICE))
        # elif args.arch=="efficientnet":
        #     # Load pretrained EfficientNet-B0
        #     model = models.efficientnet_b0(pretrained=False)
        #     # Replace the classifier head
        #     if args.dataset=="cifar10":
        #         model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
        #     elif args.dataset=="imagenet":
        #         model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        #     bias_models.append(model.to(DEVICE))
            
    # if model_type=="efficientnet":
    #     del model
    

    
    
    # --- 7. Training and Validation Loop ---
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    if args.devil:
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

    else:

        # For classical path, we train each bias model on the same spurious training data
        model_train_loader = train_loader_spurious
        model_save_path = os.path.join(args.bias_model_path, f"bias_model_original_best.pth")
        
        # Classical path — unchanged
        outputs = train_model(net, optim, criterion,
                              train_loader=train_loader_spurious,
                              val_loader=train_loader_new,
                              args=args, epochs=args.max_epoch_s,
                              log_test_results=log_test_results,
                              test_loader=test_loader,
                              test_criterion=test_criterion)

        if return_loaders:
            return net, outputs, (train_loader_new, train_loader_spurious)
        return net, outputs, None
    
def train_batch_model(train_loaders, sliced_data_indices, args,
                      val_loader, test_loader=None):
    """
    Train a single model with minibatch SGD aggregating and shuffling the sliced data indices - Updated with val loader
    """
    net = get_net(args, pretrained=False)
    optim = get_optim(net, args, model_type='pretrain')
    criterion = get_criterion(args)
    test_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    indices = np.hstack(sliced_data_indices)
    heading = f'Training on aggregated slices'
    print('-' * len(heading))
    print(heading)
    sliced_val_loader = val_loader
    sliced_train_sampler = SubsetRandomSampler(indices)
    sliced_train_loader = DataLoader(train_loaders.dataset,
                                     batch_size=args.bs_trn,
                                     sampler=sliced_train_sampler,
                                     num_workers=args.num_workers)
    args.model_type = 'mb_slice'
    train_model(net, optim, criterion, sliced_train_loader,
                sliced_val_loader, args, 0, args.max_epoch,
                True, test_loader, test_criterion)
    return net
