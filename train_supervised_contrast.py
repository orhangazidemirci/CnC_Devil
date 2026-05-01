"""
Correct-n-Contrast main script
"""

import os
from sched import scheduler
import sys
import copy
import argparse
import importlib

import torch
import torch.nn.functional as f
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
# Data
from torch.utils.data import DataLoader, SequentialSampler, SubsetRandomSampler
from datasets import train_val_split, get_resampled_indices, get_resampled_set, initialize_data
# Logging and training
from train import train_model, test_model
from evaluate import evaluate_model, run_final_evaluation
# , update_contrastive_experiment_name
from utils import free_gpu, print_header
from utils import init_experiment, init_args, update_args
from utils.logging import Logger, log_args, summarize_acc, initialize_csv_metrics, log_data
from utils.visualize import plot_confusion, plot_data_batch
from utils.metrics import compute_resampled_mutual_info, compute_mutual_info_by_slice
# Model
from network import get_net, get_optim, get_criterion, load_pretrained_model, save_checkpoint
from network import get_output, backprop_, get_bert_scheduler, _get_linear_schedule_with_warmup
# U-MAPS
from activations import visualize_activations
# Contrastive
from contrastive_supervised_loader import prepare_contrastive_points, load_contrastive_data, adjust_num_pos_neg_
from contrastive_network import DEFAULT_WEIGHTS, ContrastiveNet, load_encoder_state_dict, compute_outputs
from contrastive_network import DevilNetLoss
from slice import compute_pseudolabels, compute_slice_indices, train_spurious_model
# Alternative slicing by UMAP clustering
from slice_rep import compute_devil_net_signals, compute_slice_indices_by_rep, combine_data_indices

import transformers
transformers.logging.set_verbosity_error()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train_epoch(model, train_loader, loss_fn, optimizer, args, epoch):
    """
    One full training epoch.

    Returns dict of mean losses for the epoch.
    """
    model.train()
    losses = {'total': [], 'ce': [], 'self': [], 'batch': []}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch in pbar:
        inputs, labels, indices = batch
        inputs  = inputs.to(args.device)
        labels  = labels.to(args.device)
        indices = indices.to(args.device)

        optimizer.zero_grad()

        # Forward — model must return (embeddings, logits)
        embeddings, logits = model(inputs)

        # CE on classifier head
        ce_loss = F.cross_entropy(logits, labels)

        # Self + batch contrastive losses
        contrastive_loss, self_loss, batch_loss = loss_fn(indices, embeddings)

        total_loss = ce_loss + contrastive_loss
        total_loss.backward()

        if getattr(args, 'clip_grad_norm', False):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses['total'].append(total_loss.item())
        losses['ce'].append(ce_loss.item())
        losses['self'].append(self_loss.item())
        losses['batch'].append(batch_loss.item())

        pbar.set_postfix({k: f'{np.mean(v):.4f}' for k, v in losses.items()})

    return {k: np.mean(v) for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Step 3 — evaluate
# Returns average accuracy and worst-group accuracy
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, args, group_labels=None):
    """
    Evaluates model on a dataloader.

    Args:
        model        : model with forward() returning (embeddings, logits)
        dataloader   : evaluation DataLoader
        args         : args namespace
        group_labels : optional np.ndarray (N,) of group ids for worst-group eval

    Returns dict with:
        avg_acc       : float — overall accuracy
        worst_group   : float — worst-group accuracy (if group_labels provided)
        group_accs    : dict  — per-group accuracy (if group_labels provided)
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_idx    = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if len(batch) == 3:
                inputs, labels, indices = batch
            else:
                inputs, labels = batch
                indices = None

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            _, logits = model(inputs)
            _, preds  = torch.max(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if indices is not None:
                all_idx.append(indices.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_acc = (all_preds == all_labels).mean()
    results = {'avg_acc': avg_acc}

    # Worst-group accuracy
    if group_labels is not None:
        all_idx = np.concatenate(all_idx) if all_idx else np.arange(len(all_labels))
        group_accs = {}
        for g in np.unique(group_labels):
            g_mask         = group_labels[all_idx] == g
            group_accs[g]  = (all_preds[g_mask] == all_labels[g_mask]).mean() \
                             if g_mask.sum() > 0 else 0.0
        results['worst_group'] = min(group_accs.values())
        results['group_accs']  = group_accs

    return results

def compute_slice_outputs(erm_models, train_loaders, args):
    """
    Compute predictions of ERM model to set up contrastive batches
    """

    sliced_outputs = {
        'angel_predictions': [],
        'devil_predictions': [],
        'angel_embeddings':  [],
        'devil_embeddings':  [],
        'targets':           [],
    }

    for data_idx, train_loader in enumerate(train_loaders):
        print_header(f'Partition {data_idx}:')

        # ✓ pass ALL models, not just data_idx
        slice_output = compute_devil_net_signals(
            bias_models=erm_models,
            dataloader=train_loader,
            data_idx=data_idx,
            args=args
,                )

        for k in sliced_outputs:
            sliced_outputs[k].append(slice_output[k])

    # concatenate across all partitions
    sliced_outputs = {k: np.concatenate(v, axis=0) for k, v in sliced_outputs.items()}

    # ✓ wrap with index tracking
    combined_dataset = IndexedDataset(
        ConcatDataset([loader.dataset for loader in train_loaders])
    )
    train_loader = DataLoader(
        combined_dataset,
        batch_size=args.bs_trn,
        shuffle=False,          # must stay False — indices must match sliced_outputs
        num_workers=args.num_workers
    )

    # build loss function
    loss_fn = DevilNetLoss(sliced_outputs, weights=DEFAULT_WEIGHTS, temperature=args.temperature)


    return sliced_outputs, train_loaders, loss_fn


def finetune_model(encoder, criterion, test_criterion, dataloaders,
                   erm_models, args):
    """
    Instead of joint training, finetune classifier
    """
    train_loaders, val_loader, test_loader = dataloaders
    model = get_net(args)
    state_dict = encoder.to(torch.device('cpu')).state_dict()
    model = load_encoder_state_dict(model, state_dict)
    args.model_type = 'finetune'
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias',
                            'backbone.fc.weight',
                            'backbone.fc.bias']:
                param.requires_grad = False
        # Extra checking
        params = list(filter(lambda p: p.requires_grad,
                             model.parameters()))
        assert len(params) == 2
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                print(name)
        args.model_type += f'-fe'

    optim = get_optim(model, args, model_type='classifier')
    erm_models.to(args.device)
    erm_models.eval()
    slice_outputs = compute_slice_outputs(erm_models,
                                          train_loaders,
                                          args)
    sliced_outputs, train_loaders, loss_fn = slice_outputs
    erm_models.to(torch.device('cpu'))
    indices = np.hstack(sliced_data_indices)
    heading = f'Finetuning on aggregated slices'
    print('-' * len(heading))
    print(heading)
    sliced_val_loader = val_loader
    sliced_train_sampler = SubsetRandomSampler(indices)
    sliced_train_loaders= DataLoader(train_loaders.dataset,
                                     batch_size=args.bs_trn,
                                     sampler=sliced_train_sampler,
                                     num_workers=args.num_workers)
    args.model_type = '2s2s_ss'
    outputs = train_model(model, optim, criterion,
                          sliced_train_loaders,
                          sliced_val_loader, args, 0,
                          args.finetune_epochs, True,
                          test_loader, test_criterion)
    model, max_robust_metrics, all_acc = outputs
    return model


def train_devil_net(model, erm_models, train_loaders, val_loader, test_loader,
                    optimizer, scheduler, args, save_activations_fn,
                    group_labels_val=None, group_labels_test=None):
    """
    Full Devil-NET training loop.

    Args:
        model             : the main model being trained (encoder + classifier)
        erm_models        : list of N pretrained ERM/bias models
        train_loaders     : list of N DataLoaders, one per partition
        val_loader        : validation DataLoader
        test_loader       : test DataLoader
        optimizer         : torch optimizer
        scheduler         : lr scheduler (or None)
        args              : args namespace — needs:
                              args.max_epoch, args.bs_trn, args.num_workers,
                              args.temperature, args.alpha, args.beta,
                              args.device, args.clip_grad_norm (optional),
                              args.hard_neg_factor (optional)
        save_activations_fn: function(model, loader, args) -> (emb, pred)
        group_labels_val  : optional np.ndarray for worst-group val eval
        group_labels_test : optional np.ndarray for worst-group test eval

    Returns:
        best_model_state : state_dict of model with best val accuracy
        history          : list of per-epoch dicts with losses + metrics
    """

    sliced_outputs, train_loader, loss_fn= train_loaders

    if args.train_encoder is not True:
        return None, []

    # ------------------------------------------------------------------
    # Compute Angel/Devil signals once before training
    # ------------------------------------------------------------------
    print_header('Devil-NET: Computing Angel/Devil signals')



    # Free ERM models from memory — no longer needed
    for i in range(len(erm_models)):
        erm_models[i].to(torch.device('cpu'))
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print_header('Devil-NET: Training')

    history          = []
    best_val_acc     = -1.0
    best_model_state = None

    for epoch in range(1, args.max_epoch + 1):

        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
        )

        # Validate
        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            args=args,
            group_labels=group_labels_val,
        )

        if scheduler is not None:
            scheduler.step()

        # Log
        epoch_log = {'epoch': epoch, **train_losses, **{f'val_{k}': v
                     for k, v in val_results.items()}}
        history.append(epoch_log)

        print(f'Epoch {epoch:3d} | '
              f'loss={train_losses["total"]:.4f} '
              f'ce={train_losses["ce"]:.4f} '
              f'self={train_losses["self"]:.4f} '
              f'batch={train_losses["batch"]:.4f} | '
              f'val_avg={val_results["avg_acc"]*100:.1f}%' +
              (f' val_worst={val_results["worst_group"]*100:.1f}%'
               if 'worst_group' in val_results else ''))

        # Save best model by val average accuracy
        # (swap to worst_group if group_labels_val is provided)
        monitor = val_results.get('worst_group', val_results['avg_acc'])
        if monitor > best_val_acc:
            best_val_acc     = monitor
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    # Final test evaluation on best model
    # ------------------------------------------------------------------
    print_header('Devil-NET: Test evaluation')
    model.load_state_dict({k: v.to(args.device)
                           for k, v in best_model_state.items()})

    test_results = evaluate(
        model=model,
        dataloader=test_loader,
        args=args,
        group_labels=group_labels_test,
    )

    print(f'Test avg accuracy:   {test_results["avg_acc"]*100:.2f}%')
    if 'worst_group' in test_results:
        print(f'Test worst-group:    {test_results["worst_group"]*100:.2f}%')
        for g, acc in test_results['group_accs'].items():
            print(f'  Group {g}: {acc*100:.2f}%')

    return best_model_state, history

def main():
    parser = argparse.ArgumentParser(description='Compare & Contrast')
    # Model
 # -------------------------------------------------------------------------
    # Devil-NET
    # -------------------------------------------------------------------------
    parser.add_argument('--devil', action='store_true', default=True,
                        help='Partition training data into N shards, train one bias model per shard')
    parser.add_argument('--num_bias_models', type=int, default=5,
                        help='Number of shard partitions / bias models')

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------
    parser.add_argument('--arch', choices=['base', 'mlp', 'cnn',
                                           'resnet50', 'resnet50_pt',
                                           'resnet34', 'resnet34_pt',
                                           'bert-base-uncased_pt'],
                        required=False, default='resnet50_pt')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dim for MLP arch only')
    parser.add_argument('--no_projection_head', default=False, action='store_true',
                        help='If True, apply contrastive loss directly on encoder output')

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='waterbirds')
    parser.add_argument('--resample_class', type=str, default='',
                        choices=['upsample', 'subsample', ''],
                        help='Resample datapoints to balance classes')

    # -------------------------------------------------------------------------
    # Batch sizes
    # -------------------------------------------------------------------------
    parser.add_argument('--bs_trn', type=int, default=128)
    parser.add_argument('--bs_val', type=int, default=128)

    # -------------------------------------------------------------------------
    # Devil-NET loss weights
    # -------------------------------------------------------------------------
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Contrastive temperature τ')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for self loss (Table 1) in L_total')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for batch contrastive loss (Table 2) in L_total')
    parser.add_argument('--hard_neg_factor', type=float, default=0.0,
                        help='Dynamic hard negative reweighting factor (0 = disabled)')

    # -------------------------------------------------------------------------
    # Encoder training
    # -------------------------------------------------------------------------
    parser.add_argument('--train_encoder', default=True, action='store_true')
    parser.add_argument('--load_encoder', type=str, default='',
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--freeze_encoder', default=False, action='store_true',
                        help='Freeze encoder layers during stage 2 training')
    parser.add_argument('--finetune_epochs', type=int, default=0)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    parser.add_argument('--optim', type=str, default='sgd',
                        choices=['AdamW', 'adam', 'sgd'])
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_c', type=float, default=-1,
                        help='Classifier weight decay (-1 = same as --weight_decay)')
    parser.add_argument('--stopping_window', type=int, default=30)
    parser.add_argument('--clip_grad_norm', default=False, action='store_true',
                        help='Clip gradient norm (recommended for BERT)')
    parser.add_argument('--lr_scheduler', type=str, default='')
    parser.add_argument('--lr_scheduler_classifier', type=str, default='')

    # -------------------------------------------------------------------------
    # Bias model (stage 1) training
    # -------------------------------------------------------------------------
    parser.add_argument('--pretrained_spurious_path', default='', type=str,
                        help='Path to pretrained bias models (skips stage 1 if set)')
    parser.add_argument('--max_epoch_s', type=int, default=1,
                        help='Epochs to train each bias model')
    parser.add_argument('--bs_trn_s', type=int, default=32,
                        help='Batch size for bias model training')
    parser.add_argument('--lr_s', type=float, default=1e-3,
                        help='Learning rate for bias models')
    parser.add_argument('--momentum_s', type=float, default=0.9)
    parser.add_argument('--weight_decay_s', type=float, default=5e-4)

    # -------------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------------
    parser.add_argument('--erm', default=False, action='store_true',
                        help='Train with balanced ERM')
    parser.add_argument('--erm_only', default=False, action='store_true',
                        help='Train with standard ERM only (no debiasing)')

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    parser.add_argument('--log_loss_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--verbose', default=False, action='store_true')

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--evaluate', default=False, action='store_true')

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Derived paths
    # -------------------------------------------------------------------------
    args.results_path   = f'./results/{args.dataset}/{args.arch}/'
    args.image_path     = f'./images/{args.dataset}/{args.arch}/'
    args.model_path     = f'./model/{args.dataset}/{args.arch}/'
    args.bias_model_path = f'./model/{args.dataset}/{args.arch}/saved_bias_models'
    args.experiment_name = (f'devil-{args.devil}_arch-{args.arch}'
                            f'_bs-{args.bs_trn}_dataset-{args.dataset}')
    args.log_path       = f'./logs/{args.dataset}/{args.experiment_name}'

    if args.weight_decay_c < 0:
        args.weight_decay_c = args.weight_decay
        
    if 'waterbirds' in args.dataset:
        if not hasattr(args, 'root_dir') or args.root_dir is None:
            args.root_dir = '../slice-and-dice-smol/datasets/data/Waterbirds/'
            
        
    # load_dataloaders, visualize_dataset = initialize_data(args)
    if args.resample_class != '':
        if args.resample_class == 'upsample':
            sample += '-rsc=u'
        elif args.resample_class == 'subsample':
            sample += '-rsc=s'

    #init_args(args)
    #update_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = (torch.device('cuda:0') if torch.cuda.is_available()
                and not args.no_cuda else torch.device('cpu'))
    if os.path.exists(args.log_path) and args.resume:
        resume = True
        mode = 'a'
    else:
        resume = False
        mode = 'w'
    logger = Logger(os.path.join(args.log_path,
                                f'log-{args.experiment_name}.txt'), mode)
    log_args(args, logger)
    sys.stdout = logger

    criterion = get_criterion(args, reduction='mean')
    test_criterion = get_criterion(args, reduction='none')

    train_loaders, val_loader, test_loader, visualize_dataset = initialize_data(args)
        # train_loaders is a list of N loaders, one per bias model partition


    if args.resample_class != '':
        resampled_indices = get_resampled_indices(dataloader=train_loaders,
                                                  args=args,
                                                  sampling=args.resample_class,
                                                  seed=args.seed)
        train_set_resampled = get_resampled_set(dataset=train_loaders.dataset,
                                                resampled_set_indices=resampled_indices,
                                                copy_dataset=True)
        train_loaders= DataLoader(train_set_resampled,
                                  batch_size=args.bs_trn,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    if args.dataset != 'civilcomments':
        log_data(train_loaders[0].dataset.dataset, 'Train dataset:')  # Subset → Waterbirds
   
        log_data(val_loader.dataset, 'Val dataset:')
        log_data(test_loader.dataset, 'Test dataset:')
    if args.evaluate is True:
        initialize_csv_metrics(args)
        assert args.load_encoder != ''
        args.checkpoint_name = args.load_encoder
        try:
            start_epoch = int(args.checkpoint_name.split(
                '-cpe=')[-1].split('-')[0])
        except:
            start_epoch = 0
        try:  # Load full model
            print(f'Loading full model...')
            model = get_net(args)
            model_state_dict = torch.load(os.path.join(args.model_path,
                                                       args.checkpoint_name))
            model_state_dict = model_state_dict['model_state_dict']
            model = load_encoder_state_dict(model, model_state_dict,
                                            contrastive_train=False)
            print(f'-> Full model loaded!')
        except Exception as e:
            print(e)
            project = not args.no_projection_head
            assert args.load_encoder != ''
            args.checkpoint_name = args.load_encoder
            start_epoch = int(args.checkpoint_name.split(
                '-cpe=')[-1].split('-')[0])
            checkpoint = torch.load(os.path.join(args.model_path,
                                                 args.checkpoint_name))
            print(f'Checkpoint loading from {args.load_encoder}!')
            print(f'- Resuming training at epoch {start_epoch}')

            encoder = ContrastiveNet(args.arch, out_dim=args.projection_dim,
                                     projection_head=project,
                                     task=args.dataset,
                                     num_classes=args.num_classes,
                                     checkpoint=checkpoint)
            classifier = copy.deepcopy(encoder.classifier)
            encoder.to(torch.device('cpu'))
            classifier.to(torch.device('cpu'))
            model = get_net(args)
            state_dict = encoder.to(torch.device('cpu')).state_dict()
            for k in list(state_dict.keys()):
                if k.startswith('fc.') and 'bert' in args.arch:
                    state_dict[f'classifier.{k[3:]}'] = state_dict[k]
                    # state_dict[k[f'classifier.{k[3:]}']] = state_dict[k]
                    del state_dict[k]

            model = load_encoder_state_dict(model, state_dict)
            try:
                model.fc = classifier
            except:
                model.classifier = classifier
        run_final_evaluation(model, test_loader, test_criterion,
                             args, epoch=start_epoch,
                             visualize_representation=True)

        print('Done training')
        print(f'- Experiment name: {args.experiment_name}')
        print_header(f'Max Robust Acc:')
        print(f'Acc: {args.max_robust_acc}')
        print(f'Epoch: {args.max_robust_epoch}')
        summarize_acc(args.max_robust_group_acc[0],
                      args.max_robust_group_acc[1])
        return

    # -------------------
    # Slice training data
    # -------------------
    if args.pretrained_spurious_path != '':
        print_header('> Loading spurious model')
        erm_models = []
        for i in range(args.num_bias_models):
            print(f'Partition {i}:')
            fpath = os.path.join(args.bias_model_path, f"bias_model_{i}_best.pth")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)

            partition_erm_models = load_pretrained_model(fpath,
                                                        args)
            partition_erm_models.eval()
            erm_models.append(partition_erm_models)  

        args.mode = 'train_spurious'
    else:
        args.mode = 'train_spurious'
        print_header('> Training spurious model')
        args.spurious_train_split = 0.99
        erm_models, outputs, _ = train_spurious_model(train_loaders, args)
    
    for i in range(args.num_bias_models):
        erm_models[i].eval()
    print(f'Pretrained model loaded from {fpath}')

    if args.train_encoder is True:
    
        slice_outputs = compute_slice_outputs(erm_models,  train_loaders, args)
        sliced_outputs, train_loaders, loss_fn = slice_outputs

        for i in range(args.num_bias_models):
            print(f'Partition {i}:')
            for _, p in erm_models[i].named_parameters():
                p = p.to(torch.device('cpu'))
                erm_models[i].to(torch.device('cpu'))


        # -------------
        # Train encoder
        # -------------
        best_state, history = train_devil_net(
            model=model,
            erm_models=erm_models,
            train_loaders=slice_outputs,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args
        )
 
if __name__ == '__main__':
    main()
