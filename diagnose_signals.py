"""
Diagnostic test for compute_devil_net_signals() with real models and data.

Checks:
  1. Per-partition signal quality (TF/TT/FF/FT breakdown)
  2. Per-partition bias ratios (actual vs target)
  3. Batch pair statistics (how many valid positive/negative pairs per batch)
  4. Combined dataset size vs expected
  5. Angel/Devil prediction agreement patterns
  6. Embedding norms and similarity distributions
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict



def diagnose_devil_net_signals(bias_models, train_loaders, sliced_outputs,
                                train_loader, loss_fn, args):
    """
    Run full diagnostics on Devil-NET signals and batch construction.
 
    Args:
        bias_models    : list of N trained bias models
        train_loaders  : list of N per-partition DataLoaders
        sliced_outputs : dict output of compute_slice_outputs
        train_loader   : combined DataLoader (after OffsetDataset)
        loss_fn        : DevilNetLoss instance
        args           : args namespace
    """
    print('\n' + '='*60)
    print('  DEVIL-NET SIGNAL DIAGNOSTICS')
    print('='*60)
 
    targets     = sliced_outputs['targets']
    angel_pred  = sliced_outputs['angel_predictions']
    devil_pred  = sliced_outputs['devil_predictions']
    angel_emb   = sliced_outputs['angel_embeddings']
    devil_emb   = sliced_outputs['devil_embeddings']
    N_total     = len(targets)
    num_classes = len(np.unique(targets))
 
    # ------------------------------------------------------------------
    # 1. Combined dataset size check
    # ------------------------------------------------------------------
    print(f'\n[1] Dataset size check')
    expected  = sum(len(l.dataset) for l in train_loaders)
    actual    = len(train_loader.dataset)
    n_batches = len(train_loader)
    print(f'  Expected (sum of partitions): {expected}')
    print(f'  Combined DataLoader size:     {actual}')
    print(f'  sliced_outputs size:          {N_total}')
    print(f'  Batches per epoch:            {n_batches}  (batch_size={args.bs_trn})')
    if actual != expected:
        print(f'  *** MISMATCH — DataLoader sees {actual}/{expected} samples ***')
    elif actual != N_total:
        print(f'  *** MISMATCH — sliced_outputs {N_total} vs loader {actual} ***')
    else:
        print(f'  OK — all sizes match')
 
    # ------------------------------------------------------------------
    # 2. Per-partition signal quality
    # ------------------------------------------------------------------
    print(f'\n[2] Per-partition signal quality')
    offset = 0
    for i, loader in enumerate(train_loaders):
        n = len(loader.dataset)
        a = angel_pred[offset:offset + n]
        d = devil_pred[offset:offset + n]
        t = targets[offset:offset + n]
 
        TF        = ((a == t) & (d != t)).mean() * 100
        TT        = ((a == t) & (d == t)).mean() * 100
        FF        = ((a != t) & (d != t)).mean() * 100
        FT        = ((a != t) & (d == t)).mean() * 100
        angel_acc = (a == t).mean() * 100
        devil_acc = (d == t).mean() * 100
 
        print(f'  Partition {i} (n={n}): '
              f'TF={TF:.1f}%  TT={TT:.1f}%  FF={FF:.1f}%  FT={FT:.1f}%  '
              f'| angel_acc={angel_acc:.1f}%  devil_acc={devil_acc:.1f}%')
        offset += n
 
    # ------------------------------------------------------------------
    # 3. Per-partition bias ratios
    # ------------------------------------------------------------------
    print(f'\n[3] Per-partition bias ratios')
    offset = 0
    for i, loader in enumerate(train_loaders):
        n      = len(loader.dataset)
        t_part = targets[offset:offset + n]
        d_part = devil_pred[offset:offset + n]
 
        class_info = []
        for c in range(num_classes):
            class_mask = t_part == c
            if class_mask.sum() == 0:
                continue
            devil_correct_c = (d_part[class_mask] == c).mean() * 100
            class_info.append(f'c{c}={devil_correct_c:.1f}%')
 
        dataset = loader.dataset
        if hasattr(dataset, 'indices') and hasattr(dataset.dataset, 'metadata_array'):
            meta   = dataset.dataset.metadata_array[dataset.indices, 0]
            groups = meta.numpy() if hasattr(meta, 'numpy') else np.array(meta)
            per_class_bias = []
            for c in range(num_classes):
                n_s  = (groups == 2 * c).sum()
                n_c  = (groups == 2 * c + 1).sum()
                bias = n_s / (n_s + n_c) if (n_s + n_c) > 0 else 0
                per_class_bias.append(f'c{c}={bias:.3f}')
            print(f'  Partition {i} (n={n}): '
                  f'actual bias [{", ".join(per_class_bias)}]  '
                  f'| devil acc per class [{", ".join(class_info)}]')
        else:
            print(f'  Partition {i} (n={n}): '
                  f'devil acc per class [{", ".join(class_info)}]')
        offset += n
 
    # ------------------------------------------------------------------
    # 4. Batch pair statistics
    # ------------------------------------------------------------------
    print(f'\n[4] Batch pair statistics (first 5 batches)')
    print(f'  (batch_loss=0 when pos_total=0 or neg_total=0)')
 
    loss_fn._to_device(torch.device('cpu'))
    batch_stats = []
 
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:
            break
 
        _, labels, indices = batch
        indices = indices.cpu()
 
        y = loss_fn.targets[indices]
        a = loss_fn.angel_pred[indices]
        d = loss_fn.devil_pred[indices]
        B = len(indices)
 
        same_class = y.unsqueeze(1) == y.unsqueeze(0)
        same_angel = a.unsqueeze(1) == a.unsqueeze(0)
        same_devil = d.unsqueeze(1) == d.unsqueeze(0)
        diff_class = ~same_class
        diag       = torch.eye(B, dtype=torch.bool)
 
        case_A = same_class & ~same_angel & ~same_devil & ~diag
        case_B = same_class &  same_angel & ~same_devil & ~diag
        case_C = same_class &  same_angel &  same_devil & ~diag
        case_D = same_class & ~same_angel &  same_devil & ~diag
        case_E = diff_class & ~same_angel &  same_devil & ~diag
        case_F = diff_class &  same_angel &  same_devil & ~diag
        case_G = diff_class & ~same_angel & ~same_devil & ~diag
        case_H = diff_class &  same_angel & ~same_devil & ~diag
 
        n_pos = (case_A.sum() + case_B.sum() + case_C.sum()).item()
        n_neg = (case_E.sum() + case_F.sum()).item()
 
        pos_w, neg_w = loss_fn._pair_weights(indices)
        valid = ((pos_w > 0).any(dim=1) & (neg_w > 0).any(dim=1)).sum()
 
        batch_stats.append({'pos': n_pos, 'neg': n_neg})
        print(f'  Batch {batch_idx} (B={B}): '
              f'A={case_A.sum():3d} B={case_B.sum():3d} C={case_C.sum():3d} '
              f'D={case_D.sum():3d} | '
              f'E={case_E.sum():3d} F={case_F.sum():3d} G={case_G.sum():3d} '
              f'H={case_H.sum():3d} | '
              f'pos={n_pos:4d} neg={n_neg:4d} | '
              f'valid_anchors={valid.item()}/{B}')
 
    if sum(s['pos'] for s in batch_stats) == 0:
        print(f'\n  *** NO POSITIVE PAIRS — batch_loss=0 expected ***')
        print(f'  same_class & ~same_angel & ~same_devil never True')
    if sum(s['neg'] for s in batch_stats) == 0:
        print(f'  *** NO NEGATIVE PAIRS ***')
        print(f'  diff_class & same_devil never True — devil too accurate')
 
    # ------------------------------------------------------------------
    # 5. Angel/Devil prediction agreement
    # ------------------------------------------------------------------
    print(f'\n[5] Angel/Devil prediction agreement')
    same_pred   = (angel_pred == devil_pred).mean() * 100
    angel_eq_gt = (angel_pred == targets).mean() * 100
    devil_eq_gt = (devil_pred == targets).mean() * 100
 
    print(f'  Angel pred == Devil pred:  {same_pred:.1f}%  '
          f'(<- high = fewer hard positives, case A)')
    print(f'  Angel pred == true label:  {angel_eq_gt:.1f}%')
    print(f'  Devil pred == true label:  {devil_eq_gt:.1f}%')
    print(f'\n  Prediction class distribution:')
    for c in np.unique(targets):
        a_c = (angel_pred == c).mean() * 100
        d_c = (devil_pred == c).mean() * 100
        t_c = (targets == c).mean() * 100
        print(f'    Class {c}: true={t_c:.1f}%  '
              f'angel_pred={a_c:.1f}%  devil_pred={d_c:.1f}%')
 
    # ------------------------------------------------------------------
    # 6. Embedding statistics
    # ------------------------------------------------------------------
    print(f'\n[6] Embedding statistics')
    angel_norms = np.linalg.norm(angel_emb, axis=1)
    devil_norms = np.linalg.norm(devil_emb, axis=1)
    a_norm      = angel_emb / (angel_norms[:, None] + 1e-8)
    d_norm      = devil_emb / (devil_norms[:, None] + 1e-8)
    cos_sim     = (a_norm * d_norm).sum(axis=1)
 
    print(f'  Angel emb norm: mean={angel_norms.mean():.2f}  '
          f'std={angel_norms.std():.2f}  '
          f'min={angel_norms.min():.2f}  max={angel_norms.max():.2f}')
    print(f'  Devil emb norm: mean={devil_norms.mean():.2f}  '
          f'std={devil_norms.std():.2f}  '
          f'min={devil_norms.min():.2f}  max={devil_norms.max():.2f}')
    print(f'  Angel-Devil cosine sim: mean={cos_sim.mean():.3f}  '
          f'std={cos_sim.std():.3f}  '
          f'min={cos_sim.min():.3f}  max={cos_sim.max():.3f}')
    if cos_sim.mean() > 0.95:
        print(f'  *** HIGH sim — angel/devil embeddings nearly identical ***')
        print(f'  -> contrastive signal will be weak')
 
    # ------------------------------------------------------------------
    # 7. Devil confusion matrix per partition
    # ------------------------------------------------------------------
    print(f'\n[7] Devil confusion matrix per partition')
    print(f'  (Spurious errors: e.g. landbird-on-water predicted as waterbird)')
    offset = 0
    for i, loader in enumerate(train_loaders):
        n      = len(loader.dataset)
        t_part = targets[offset:offset + n]
        d_part = devil_pred[offset:offset + n]
 
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for true_c in range(num_classes):
            for pred_c in range(num_classes):
                conf[true_c, pred_c] = ((t_part == true_c) &
                                        (d_part == pred_c)).sum()
 
        print(f'  Partition {i}:')
        header = '         ' + ''.join(f'  pred_{c}' for c in range(num_classes))
        print(f'  {header}')
        for true_c in range(num_classes):
            row = ''.join(f'  {conf[true_c, c]:6d}' for c in range(num_classes))
            print(f'    true_{true_c}{row}')
        for true_c in range(num_classes):
            errors = conf[true_c].sum() - conf[true_c, true_c]
            if errors > 0:
                rate = errors / conf[true_c].sum() * 100
                print(f'    -> class {true_c}: {errors} errors ({rate:.1f}% error rate)')
        offset += n
 
    # ------------------------------------------------------------------
    # 8. Per-class angel/devil accuracy
    # ------------------------------------------------------------------
    print(f'\n[8] Per-class angel/devil accuracy')
    print(f'  (Frozen encoder: devil should fail more on minority-group samples)')
    for c in range(num_classes):
        mask        = targets == c
        angel_acc_c = (angel_pred[mask] == targets[mask]).mean() * 100
        devil_acc_c = (devil_pred[mask] == targets[mask]).mean() * 100
        n_c         = mask.sum()
        print(f'  Class {c} (n={n_c}): '
              f'angel_acc={angel_acc_c:.1f}%  devil_acc={devil_acc_c:.1f}%')
 
    # ------------------------------------------------------------------
    # 9. Devil embedding similarity between classes
    # ------------------------------------------------------------------
    print(f'\n[9] Devil embedding similarity between classes')
    print(f'  (High cross-class sim = spurious grouping in embedding space)')
    d_norm_all = devil_emb / (np.linalg.norm(devil_emb, axis=1, keepdims=True) + 1e-8)
    for c0 in range(num_classes):
        for c1 in range(c0 + 1, num_classes):
            mask_c0   = targets == c0
            mask_c1   = targets == c1
            sim_cross = (d_norm_all[mask_c0] @ d_norm_all[mask_c1].T).mean()
            sim_same  = (d_norm_all[mask_c0] @ d_norm_all[mask_c0].T)
            np.fill_diagonal(sim_same, np.nan)
            sim_same = np.nanmean(sim_same)
            print(f'  sim(devil_c{c0}, devil_c{c1}) = {sim_cross:.3f}  '
                  f'| sim(devil_c{c0}, devil_c{c0}) = {sim_same:.3f}')
            if sim_cross > 0.8 * sim_same:
                print(f'  -> Cross-class similarity high relative to within-class')
                print(f'     Devil is NOT separating classes in embedding space')
 
    # ------------------------------------------------------------------
    # 10. Angel fallback rate
    # ------------------------------------------------------------------
    print(f'\n[10] Angel fallback rate')
    print(f'  (Samples where no Angel was correct -> mean embedding fallback)')
    fallback_mask = angel_pred != targets
    fallback_rate = fallback_mask.mean() * 100
    print(f'  Fallback samples: {fallback_mask.sum()} / {N_total} '
          f'({fallback_rate:.1f}%)')
    if fallback_rate > 10:
        print(f'  -> High fallback: mean embedding used for many samples')
        print(f'     These have weak/noisy angel signal')
 
    # ------------------------------------------------------------------
    # 11. Batch index range (OffsetDataset verification)
    # ------------------------------------------------------------------
    print(f'\n[11] Batch index range (OffsetDataset verification)')
    expected_boundaries = []
    off = 0
    for loader in train_loaders:
        expected_boundaries.append((off, off + len(loader.dataset) - 1))
        off += len(loader.dataset)
    print(f'  Expected partition boundaries: {expected_boundaries}')
 
    all_indices_seen = []
    for batch_idx, batch in enumerate(train_loader):
        _, _, indices = batch
        all_indices_seen.extend(indices.cpu().numpy().tolist())
        if batch_idx >= 9:
            break
 
    all_indices_seen = np.array(all_indices_seen)
    oob = (all_indices_seen >= N_total).sum()
    print(f'  First 10 batches index range: '
          f'{all_indices_seen.min()} - {all_indices_seen.max()}')
    print(f'  Out-of-bounds indices: {oob} (should be 0)')
    if oob > 0:
        print(f'  *** OUT OF BOUNDS — OffsetDataset bug ***')
    else:
        print(f'  OK — all indices within bounds')
 
    print('\n' + '='*60)
    print('  END DIAGNOSTICS')
    print('='*60 + '\n')
 


# ---------------------------------------------------------------------------
# Call this right after compute_slice_outputs in your main():
# ---------------------------------------------------------------------------
#
# train_loader, loss_fn = compute_slice_outputs(erm_models, train_loaders, args)
#
# diagnose_devil_net_signals(
#     bias_models=erm_models,
#     train_loaders=train_loaders,
#     sliced_outputs=sliced_outputs,   # need to return this from compute_slice_outputs
#     train_loader=train_loader,
#     loss_fn=loss_fn,
#     args=args,
# )