"""
Devil-NET: Batch Sampler and Loss Function
==========================================

Pipeline:
  1. compute_devil_net_signals()  — precompute Angel/Devil embeddings + preds (once, before training)
  2. DevilNetLoss                 — classifies each sample/pair via Table 1 & Table 2,
                                    returns weighted self + batch contrastive loss
  3. train_step()                 — integrates into standard training loop

Loss overview:
  L_total = L_CE  +  α · L_self  +  β · L_batch

  L_self  (Table 1): per-sample, pushes f_enc(x) toward Angel_enc(x) and away from Devil_enc(x)
  L_batch (Table 2): per-pair,   SupCon-style over Angel positives and Devil negatives
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode
from activations import save_activations


# ---------------------------------------------------------------------------
# Step 1 — Precompute signals (run once before training)
# ---------------------------------------------------------------------------

def get_targets_all(dataset):
    """Unwrap Subset to get targets_all"""
    if hasattr(dataset, 'dataset'):  # Subset
        return dataset.dataset.targets_all
    return dataset.targets_all


def compute_devil_net_signals(bias_models, dataloader, data_idx, args):
    """
    Returns per-sample Angel/Devil predictions and embeddings
    for use in the Devil-NET contrastive loss.
    """
    targets = get_targets_all(dataloader.dataset)['target']
    n_samples = len(targets)

    # --- Devil: own model (seen this shard) ---
    devil_embeddings, devil_predictions = save_activations(
        bias_models[data_idx], dataloader, args)

    # --- Angel: aggregate over all unseen models ---
    angel_embeddings_all = []
    angel_predictions_all = []

    for model_idx, model in enumerate(bias_models):
        if model_idx == data_idx:
            continue                          # skip Devil
        emb, pred = save_activations(model, dataloader, args)
        angel_embeddings_all.append(emb)
        angel_predictions_all.append(pred)

    # majority vote for Angel prediction
    angel_predictions_stack = np.stack(angel_predictions_all, axis=1)  # (N, n_angels)

# --- Best Angel: per sample, pick the Angel with highest confidence on correct class ---
    # Requires save_activations to also return logits — if not available, falls back to first-correct

    n_samples = len(targets)
    emb_dim   = angel_embeddings_all[0].shape[1]

    angel_embeddings  = np.zeros((n_samples, emb_dim), dtype=np.float32)
    angel_predictions = np.zeros(n_samples, dtype=np.int64)

    for sample_idx in range(n_samples):
        best_emb   = None
        best_pred  = None
        best_conf  = -1.0

        for emb, pred in zip(angel_embeddings_all, angel_predictions_all):
            if pred[sample_idx] == targets[sample_idx]:
                # Use L2 norm of embedding as a proxy for confidence
                # (higher norm = more activated = more certain)
                conf = np.linalg.norm(emb[sample_idx])
                if conf > best_conf:
                    best_conf = conf
                    best_emb  = emb[sample_idx]
                    best_pred = pred[sample_idx]

        if best_emb is not None:
            angel_embeddings[sample_idx]  = best_emb
            angel_predictions[sample_idx] = best_pred
        else:
            # No Angel got this sample right — fall back to mean embedding
            angel_embeddings[sample_idx]  = np.mean(
                [e[sample_idx] for e in angel_embeddings_all], axis=0)
            angel_predictions[sample_idx] = angel_predictions_all[0][sample_idx]
            
    return {
        'angel_predictions': angel_predictions,   # â per sample
        'devil_predictions': devil_predictions,   # d̂ per sample
        'angel_embeddings':  angel_embeddings,    # z_angel per sample
        'devil_embeddings':  devil_embeddings,    # z_devil per sample
        'targets':           targets,
    }




# ---------------------------------------------------------------------------
# Step 2 — Loss module
# ---------------------------------------------------------------------------

# Weight keys and their Table 1 / Table 2 roles:
#
# Table 1 (self):
#   w1  — TT: Angel_pos only
#   w2  — TF: Angel_pos component
#   w3  — TF: Devil_neg component
#   w4  — FF: Devil_neg
#   w5  — FT: Devil_neg (conservative, w5 << w4)
#
# Table 2 (batch):
#   w_A — hard positive  (same class, â′≠â, d̂′≠d̂)
#   w_B — soft positive  (same class, â′=â,  d̂′≠d̂)
#   w_C — easy positive  (same class, â′=â,  d̂′=d̂)
#   w_E — hard negative  (diff class, â′≠â,  d̂′=d̂)
#   w_F — soft negative  (diff class, â′=â,  d̂′=d̂)

DEFAULT_WEIGHTS = {
    # self
    'w1': 0.7,
    'w2': 1.0,
    'w3': 1.0,
    'w4': 0.5,
    'w5': 0.1,
    # batch positives (hard > soft > easy)
    'w_A': 1.0,
    'w_B': 0.7,
    'w_C': 0.3,
    # batch negatives (hard > soft)
    'w_E': 1.0,
    'w_F': 0.7,
}


class DevilNetLoss(nn.Module):
    """
    Computes the Devil-NET contrastive loss for a batch.

    Usage:
        loss_fn = DevilNetLoss(signals, weights=DEFAULT_WEIGHTS, temperature=0.07)
        self_loss, batch_loss = loss_fn(batch_indices, f_embeddings)
        total = ce_loss + alpha * self_loss + beta * batch_loss
    """

    def __init__(self, signals: dict, weights: dict = None, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
        self.w   = DEFAULT_WEIGHTS.copy()
        if weights:
            self.w.update(weights)

        # Store signals as tensors (moved to device at first forward call)
        self.angel_pred = torch.from_numpy(signals['angel_predictions'])  # (N,) int
        self.devil_pred = torch.from_numpy(signals['devil_predictions'])  # (N,) int
        self.targets    = torch.from_numpy(signals['targets'])            # (N,) int

        # Normalize embeddings once
        a_emb = torch.from_numpy(signals['angel_embeddings'])            # (N, D)
        d_emb = torch.from_numpy(signals['devil_embeddings'])            # (N, D)
        self.angel_emb  = F.normalize(a_emb, dim=1)
        self.devil_emb  = F.normalize(d_emb, dim=1)

        self._device_set = False

    def _to_device(self, device):
        if not self._device_set:
            self.angel_pred = self.angel_pred.to(device)
            self.devil_pred = self.devil_pred.to(device)
            self.targets    = self.targets.to(device)
            self.angel_emb  = self.angel_emb.to(device)
            self.devil_emb  = self.devil_emb.to(device)
            self._device_set = True

    # ------------------------------------------------------------------
    # Table 1: self-signal classification
    # ------------------------------------------------------------------

    def _self_cases(self, idx: torch.Tensor):
        """
        Classify each sample in idx by Table 1 case.
        Returns masks for TT, TF, FF, FT.
        idx: (B,) dataset indices
        """
        angel_correct = self.angel_pred[idx] == self.targets[idx]   # (B,) bool
        devil_correct = self.devil_pred[idx]  == self.targets[idx]   # (B,) bool

        TT = angel_correct &  devil_correct
        TF = angel_correct & ~devil_correct
        FF = ~angel_correct & ~devil_correct
        FT = ~angel_correct &  devil_correct
        return TT, TF, FF, FT

    def _compute_self_loss(self, idx: torch.Tensor, z: torch.Tensor):
        """
        Table 1 loss for the batch.
        idx: (B,) dataset indices
        z:   (B, D) L2-normalized f_enc embeddings
        """

        self._to_device(z.device)
        w = self.w
        angel_correct = self.angel_pred[idx] == self.targets[idx]

        w = self.w
        TT, TF, FF, FT = self._self_cases(idx)

        z_a = self.angel_emb[idx]   # (B, D) — already normalized
        z_d = self.devil_emb[idx]   # (B, D)

        # cosine similarity, one scalar per sample
        angel_sim = (z * z_a).sum(dim=1)   # (B,)
        devil_sim  = (z * z_d).sum(dim=1)  # (B,)

        # minimize: -sim for angel_pos, +sim for devil_neg
        loss = torch.zeros(len(idx), device=z.device)
        loss[TT] = 0 #-w['w1'] * angel_sim[TT]
        loss[TF] = -w['w2'] * angel_sim[TF] + w['w3'] * devil_sim[TF]
        loss[FF] =  w['w4'] * devil_sim[FF]
        loss[FT] =  w['w5'] * devil_sim[FT]

        return loss.mean()

    # ------------------------------------------------------------------
    # Table 2: batch-pair classification
    # ------------------------------------------------------------------

    def _pair_weights(self, idx: torch.Tensor):
        """
        For each pair (i, j) in the batch, determine Table 2 case and weights.

        Returns:
            pos_mask:    (B, B) bool  — pairs that are positives
            neg_mask:    (B, B) bool  — pairs that are negatives
            pos_weights: (B, B) float — weight for each positive pair
            neg_weights: (B, B) float — weight for each negative pair
        """
        B   = len(idx)
        w   = self.w
        y   = self.targets[idx]    # (B,)
        a   = self.angel_pred[idx] # (B,)
        d   = self.devil_pred[idx] # (B,)

        # Pairwise comparison matrices (B, B)
        same_class = y.unsqueeze(1) == y.unsqueeze(0)       # (B, B)
        same_angel = a.unsqueeze(1) == a.unsqueeze(0)       # (B, B)
        same_devil = d.unsqueeze(1) == d.unsqueeze(0)       # (B, B)

        diff_angel = ~same_angel
        diff_devil = ~same_devil

        # --- Positives (Table 2, same class) ---
        case_A = same_class & diff_angel & diff_devil   # hard positive
        case_B = same_class & same_angel & diff_devil   # soft positive
        case_C = same_class & same_angel & same_devil   # easy positive
        # case D (same_class & diff_angel & same_devil) → skip (conflict)

        pos_mask = case_A | case_B | case_C

        pos_weights = torch.zeros(B, B, device=y.device)
        pos_weights[case_A] = w['w_A']
        pos_weights[case_B] = w['w_B']
        pos_weights[case_C] = w['w_C']

        # --- Negatives (Table 2, different class) ---
        diff_class = ~same_class
        case_E = diff_class & diff_angel & same_devil   # hard negative
        case_F = diff_class & same_angel & same_devil   # soft negative
        # case G (diff_class & diff_angel & diff_devil) → skip (easy negative)
        # case H (diff_class & same_angel & diff_devil) → skip (noise)

        neg_mask = case_E | case_F

        neg_weights = torch.zeros(B, B, device=y.device)
        neg_weights[case_E] = w['w_E']
        neg_weights[case_F] = w['w_F']

        # Exclude self-pairs
        diag = torch.eye(B, dtype=torch.bool, device=y.device)
        pos_mask[diag] = False
        neg_mask[diag] = False

        return pos_mask, neg_mask, pos_weights, neg_weights

    def _compute_batch_loss(self, idx: torch.Tensor, z: torch.Tensor):
        """
        Table 2 SupCon-style loss.
        Numerator:   weighted Angel embeddings of positive partners
        Denominator: numerator + weighted Devil embeddings of negative partners

        idx: (B,) dataset indices
        z:   (B, D) L2-normalized f_enc embeddings
        """
        B = len(idx)

        pos_mask, neg_mask, pos_w, neg_w = self._pair_weights(idx)

        z_angel_batch = self.angel_emb[idx]  # (B, D)  positives
        z_devil_batch = self.devil_emb[idx]  # (B, D)  negatives

        # Similarity: anchor z vs Angel of partner (positives)
        # sim_pos[i, j] = sim(z_i, z_angel_j)
        sim_pos = torch.mm(z, z_angel_batch.t()) / self.tau   # (B, B)

        # Similarity: anchor z vs Devil of partner (negatives)
        # sim_neg[i, j] = sim(z_i, z_devil_j)
        sim_neg = torch.mm(z, z_devil_batch.t()) / self.tau   # (B, B)

        loss = torch.zeros(B, device=z.device)
        valid = torch.zeros(B, dtype=torch.bool, device=z.device)

        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue  # anchor has no valid positive or negative — skip

            # weighted numerator (sum over positives)
            w_pos   = pos_w[i][pos_indices]                          # (n_pos,)
            exp_pos = w_pos * sim_pos[i][pos_indices].exp()          # (n_pos,)

            # weighted denominator (positives + negatives)
            w_neg   = neg_w[i][neg_indices]                          # (n_neg,)
            exp_neg = w_neg * sim_neg[i][neg_indices].exp()          # (n_neg,)

            numerator   = exp_pos.sum()
            denominator = numerator + exp_neg.sum()

            loss[i] = -torch.log(numerator / (denominator + 1e-8))
            valid[i] = True

        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        return loss[valid].mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_indices: torch.Tensor, f_embeddings: torch.Tensor):
        """
        Args:
            batch_indices : (B,) LongTensor — dataset indices for this batch
            f_embeddings  : (B, D) FloatTensor — raw f_enc outputs (will be normalized)

        Returns:
            self_loss  : scalar — Table 1 loss
            batch_loss : scalar — Table 2 loss
        """
        self._to_device(f_embeddings.device)

        # Normalize anchor embeddings once
        z = F.normalize(f_embeddings, dim=1)   # (B, D)

        self_loss  = self._compute_self_loss(batch_indices, z)
        batch_loss = self._compute_batch_loss(batch_indices, z)

        return self_loss, batch_loss


# ---------------------------------------------------------------------------
# Step 3 — Training loop integration
# ---------------------------------------------------------------------------

def train_step(model, batch_indices, batch_images, batch_labels,
               loss_fn: DevilNetLoss, optimizer,
               alpha: float = 1.0, beta: float = 1.0):
    """
    One training step for Devil-NET.

    Args:
        model         : the model being trained; forward() must return (embeddings, logits)
        batch_indices : (B,) LongTensor — dataset indices matching the precomputed signals
        batch_images  : (B, C, H, W) input images
        batch_labels  : (B,) true class labels
        loss_fn       : DevilNetLoss instance (holds precomputed signals)
        optimizer     : torch optimizer
        alpha         : weight for self loss
        beta          : weight for batch contrastive loss

    Returns:
        dict with individual and total loss values
    """
    optimizer.zero_grad()

    embeddings, logits = model(batch_images)   # (B, D), (B, C)

    # Standard cross-entropy on classifier head
    ce_loss = F.cross_entropy(logits, batch_labels)

    # Self + batch contrastive losses
    self_loss, batch_loss = loss_fn(batch_indices, embeddings)

    total_loss = ce_loss + alpha * self_loss + beta * batch_loss
    total_loss.backward()
    optimizer.step()

    return {
        'total':  total_loss.item(),
        'ce':     ce_loss.item(),
        'self':   self_loss.item(),
        'batch':  batch_loss.item(),
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Mock signals (replace with compute_devil_net_signals output)
    N, D, C = 1000, 128, 10
    signals = {
        'angel_predictions': np.random.randint(0, C, N),
        'devil_predictions':  np.random.randint(0, C, N),
        'angel_embeddings':   np.random.randn(N, D).astype(np.float32),
        'devil_embeddings':   np.random.randn(N, D).astype(np.float32),
        'targets':            np.random.randint(0, C, N),
    }

    loss_fn = DevilNetLoss(signals, weights=DEFAULT_WEIGHTS, temperature=0.07)

    # Mock batch
    B = 32
    batch_indices  = torch.randint(0, N, (B,))
    f_embeddings   = torch.randn(B, D)

    self_loss, batch_loss = loss_fn(batch_indices, f_embeddings)
    print(f"self_loss:  {self_loss.item():.4f}")
    print(f"batch_loss: {batch_loss.item():.4f}")
