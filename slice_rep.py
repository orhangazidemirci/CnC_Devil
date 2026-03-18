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
from slice import compute_pseudolabels, train_spurious_model, compute_slice_indices
from utils.logging import log_data, initialize_csv_metrics
from train import train_model, test_model, train, evaluate
from utils import print_header, init_experiment

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

    

def compute_slice_indices_by_rep(bias_models, dataloaders,
                                 cluster_umap=True, 
                                 umap_components=2,
                                 cluster_method='kmeans',
                                 args=None,
                                 visualize=False,
                                 cmap='tab10'):
    
    sliced_data_indices = []
    sliced_data_correct = []
    sliced_data_losses  = []

    # Compute cumulative offset for each partition
    partition_sizes = [len(dataloader.dataset) for dataloader in dataloaders]
    partition_offsets = np.cumsum([0] + partition_sizes[:-1])  # [0, len(D0), len(D0)+len(D1), ...]
    
    for data_idx, dataloader in enumerate(dataloaders):
        
        offset = partition_offsets[data_idx]  # index shift for this partition
      #  targets = dataloader.dataset.targets_all['target']
        targets = get_targets_all(dataloader.dataset)['target']

        all_predictions = {}
        all_embeddings = {}
        
        # Collect predictions from all unseen models
        for model_idx, model in enumerate(bias_models):
            if data_idx == model_idx:
                continue
            embeddings, predictions = save_activations(model, dataloader, args, model_idx)
            all_predictions[model_idx] = predictions
            all_embeddings[model_idx] = embeddings
        
        # Get own model embeddings → negatives
        own_embeddings, own_predictions = save_activations(
            bias_models[data_idx], dataloader, args, data_idx)
    
        # Per-sample best unseen model
        n_samples = len(targets)
        best_model_per_sample = np.full(n_samples, -1)
        
        for sample_idx in range(n_samples):
            for model_idx, preds in all_predictions.items():
                if preds[sample_idx] == targets[sample_idx]:
                    best_model_per_sample[sample_idx] = model_idx
                    break
        
        # Build best embeddings (anchors/positives)
        best_embeddings = np.zeros((n_samples, own_embeddings.shape[1]))
        for sample_idx in range(n_samples):
            m = best_model_per_sample[sample_idx]
            if m != -1:
                best_embeddings[sample_idx] = all_embeddings[m][sample_idx]
            else:
                best_embeddings[sample_idx] = all_embeddings[list(all_embeddings.keys())[0]][sample_idx]
        
        # Cluster on best embeddings
        if cluster_umap:
            umap_ = umap.UMAP(random_state=args.seed, n_components=umap_components)
            X_best = umap_.fit_transform(best_embeddings)
        else:
            X_best = best_embeddings
            
        # Cluster on own embeddings
        if cluster_umap:
            umap_ = umap.UMAP(random_state=args.seed, n_components=umap_components)
            X_own = umap_.fit_transform(own_embeddings)
        else:
            X_own = own_embeddings
    
        n_clusters = args.num_classes
        
        if cluster_method == 'kmeans':
            clusterer_best = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            cluster_labels_best = clusterer_best.fit_predict(X_best)
            means_best = clusterer_best.cluster_centers_
            clusterer_own = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
            cluster_labels_own = clusterer_own.fit_predict(X_own)
            means_own = clusterer_own.cluster_centers_
        elif cluster_method == 'gmm':
            clusterer_best = GaussianMixture(n_components=n_clusters, random_state=args.seed, n_init=10)
            cluster_labels_best = clusterer_best.fit_predict(X_best)
            means_best = clusterer_best.means_
            clusterer_own = GaussianMixture(n_components=n_clusters, random_state=args.seed, n_init=10)
            cluster_labels_own = clusterer_own.fit_predict(X_own)
            means_own = clusterer_own.means_
        else:
            raise NotImplementedError

        cluster_labels_best, cluster_correct_best = compute_cluster_assignment(
            cluster_labels_best, dataloader)
        cluster_labels_own, cluster_correct_own = compute_cluster_assignment(
            cluster_labels_own, dataloader)
            
        # Per-sample correctness from own model
        own_correct = (own_predictions == targets).astype(int)
        
        for label in np.unique(cluster_labels_best):
            group = np.where(cluster_labels_best == label)[0]
            # Shift indices by partition offset
            sliced_data_indices.append(group + offset)
            sliced_data_correct.append(own_correct[group])
            center = means_best[label]
            l2_dist = np.linalg.norm(X_best[group] - center, axis=1)
            sliced_data_losses.append(l2_dist)
                    
        if visualize:
            targets_all = get_targets_all(dataloader.dataset)
            subset_indices = dataloader.dataset.indices if hasattr(dataloader.dataset, 'indices') else np.arange(len(dataloader.dataset))

            # Compute own clustering only for visualization
            if cluster_umap:
                umap_ = umap.UMAP(random_state=args.seed, n_components=umap_components)
                X_own = umap_.fit_transform(own_embeddings)
            else:
                X_own = own_embeddings

            if cluster_method == 'kmeans':
                clusterer_own = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
                cluster_labels_own = clusterer_own.fit_predict(X_own)
            elif cluster_method == 'gmm':
                clusterer_own = GaussianMixture(n_components=n_clusters, random_state=args.seed, n_init=10)
                cluster_labels_own = clusterer_own.fit_predict(X_own)
            cluster_labels_own, _ = compute_cluster_assignment(cluster_labels_own, dataloader)

            for space_name, X_vis, cluster_labels_vis in [
                ('best', X_best, cluster_labels_best),
                ('own',  X_own,  cluster_labels_own)
            ]:
                colors = np.array(cluster_labels_vis).astype(int)
                num_colors = len(np.unique(colors))
                plt.scatter(X_vis[:, 0], X_vis[:, 1], c=colors, s=1.0,
                            cmap=plt.cm.get_cmap(cmap, num_colors))
                plt.colorbar(ticks=np.unique(colors))
                fpath = os.path.join(args.image_path,
                                    f'umap-init_slice-cr-{space_name}-d{data_idx}-{args.experiment_name}.png')
                fpath = os.path.abspath(fpath)
                if len(fpath) > 260:
                    fpath = '\\\\\\\\?\\\\' + fpath
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                plt.savefig(fname=fpath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f'Saved UMAP ({space_name}) to {fpath}!')
                for target_type in ['target', 'spurious']:
                    colors = np.array(targets_all[target_type])[subset_indices].astype(int)
                    num_colors = len(np.unique(colors))
                    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=colors, s=1.0,
                                cmap=plt.cm.get_cmap(cmap, num_colors))
                    plt.colorbar(ticks=np.unique(colors))
                    t = f'{target_type[0]}{target_type[-1]}'
                    fpath = os.path.join(args.image_path,
                                        f'umap-init_slice-{t}-{space_name}-d{data_idx}-{args.experiment_name}.png')
                    fpath = os.path.abspath(fpath)
                    if len(fpath) > 260:
                        fpath = '\\\\\\\\?\\\\' + fpath
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    plt.savefig(fname=fpath, dpi=300, bbox_inches='tight')
                    print(f'Saved UMAP ({space_name}, {target_type}) to {fpath}!')
            plt.close()
    
    return sliced_data_indices, sliced_data_correct, sliced_data_losses


def compute_cluster_assignment(cluster_labels, dataloader):
    all_correct = []
    all_correct_by_datapoint = []
    #all_targets = dataloader.dataset.targets_all['target']
    all_targets = get_targets_all(dataloader.dataset)['target']

    # This permutations thing is gross - not actually Hungarian here?
    cluster_label_permute = list(permutations(np.unique(cluster_labels)))
    for cluster_map in cluster_label_permute:
        preds = np.vectorize(cluster_map.__getitem__)(cluster_labels)
        all_targets
        correct = (preds == all_targets)
        all_correct.append(correct.sum())
        all_correct_by_datapoint.append(correct)
    all_correct = np.array(all_correct) / len(all_targets)
    
    # Find best assignment
    best_map = cluster_label_permute[np.argmax(all_correct)]
    cluster_labels = np.vectorize(best_map.__getitem__)(cluster_labels)
    cluster_correct = all_correct_by_datapoint[
        np.argmax(all_correct)].astype(int)
    return cluster_labels, cluster_correct


def combine_data_indices(sliced_data_indices, sliced_data_correct):
    """
    If computing slices from both the ERM model's predictions and 
    representation clustering, use to consolidate into single list of slice indices
    Args:
    - sliced_data_indices (np.array[][]): List of list of sliced indices from ERM and representation clustering, 
                                          e.g. [sliced_indices_pred, sliced_indices_rep],
                                          where sliced_indices_pred = [indices_with_pred_val_1, ... indices_with_pred_val_N]
    - sliced_data_correct (np.array[][]): Same as above, but if the prediction / cluster assignment was correct
    Returns:
    - total_sliced_data_indices (np.array[]): List of combined data indices per slice
    - total_sliced_data_correct (np.array[]): List of combined per-data losses per slice
    """
    sliced_data_indices, sliced_data_indices_ = sliced_data_indices
    sliced_data_correct, sliced_data_correct_ = sliced_data_correct
    total_sliced_data_indices = [[i] for i in sliced_data_indices]
    total_sliced_data_correct = [[c] for c in sliced_data_correct]
    for slice_ix, indices in enumerate(sliced_data_indices_):
        incorrect_ix = np.where(sliced_data_correct_[slice_ix] == 0)[0]
        incorrect_ix_rep = np.where(total_sliced_data_correct[slice_ix][0] == 0)[0]
        incorrect_indices = []
        # This may be slow?
        for i in indices[incorrect_ix]:
            if i not in total_sliced_data_indices[slice_ix][0][incorrect_ix_rep]:
                incorrect_indices.append(i)
        total_sliced_data_indices[slice_ix].append(np.array(incorrect_indices).astype(int))
        total_sliced_data_correct[slice_ix].append(np.zeros(len(incorrect_indices)))
        total_sliced_data_indices[slice_ix] = np.concatenate(total_sliced_data_indices[slice_ix])
        total_sliced_data_correct[slice_ix] = np.concatenate(total_sliced_data_correct[slice_ix])
    return total_sliced_data_indices, total_sliced_data_correct
    