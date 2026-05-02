"""
Contrastive network architecture, loss, and functions
"""

import torch
import torch.nn as nn
import torchvision.models as models

from copy import deepcopy
from transformers import BertForSequenceClassification, BertConfig
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import free_gpu
from network import CNN, MLP, get_output  

from resnet import *


def load_encoder_state_dict(model, state_dict, contrastive_train=False):
    # Remove 'backbone' prefix for loading into model
    if contrastive_train:
        log = model.load_state_dict(state_dict, strict=False)
        for k in list(state_dict.keys()):
            print(k)
    else:
        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):  
                # Corrected for CNN
                if k.startswith('backbone.fc1') or k.startswith('backbone.fc2'):
                    state_dict[k[len("backbone."):]] = state_dict[k]
                # Should also be corrected for BERT models
                elif (k.startswith('backbone.fc') or
                      k.startswith('backbone.classifier')):
                    pass
                else:
                    state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
        log = model.load_state_dict(state_dict, strict=False)
    print(f'log.missing_keys: {log.missing_keys}')
    return model
    
    
class ContrastiveNet(nn.Module):

    def __init__(self, base_model, out_dim, projection_head=True,
                 task=None, num_classes=None, checkpoint=None):
        super(ContrastiveNet, self).__init__()
        self.task = task
        self.num_classes = num_classes
        self.checkpoint = checkpoint
        
        if base_model[-3:] == '_pt':
            self.pretrained = True
            base_model = base_model[:-3]
        else:
            self.pretrained = False
        print(f'Loading with {base_model} backbone')
        self.base_model = base_model
        # Also adds classifier, retreivable with self.classifier
        self.backbone = self.init_basemodel(base_model)
        self.projection_head = projection_head
        self.backbone = self.init_projection_head(self.backbone, 
                                                  out_dim,
                                                  project=projection_head)
        
    def init_basemodel(self, model_name):
        try:
            if 'resnet50' in model_name:
                # model_name = 'resnet50'
                model = resnet50(pretrained=self.pretrained)
                d = model.fc.in_features
                model.fc = nn.Linear(d, self.num_classes)
                self.activation_layer = 'backbone.avgpool'
                
            elif 'cnn' in model_name:
                model = CNN(num_classes=self.num_classes)
                self.activation_layer = torch.nn.ReLU
                
            elif 'mlp' in model_name:
                model = MLP(num_classes=self.num_classes, 
                            hidden_dim=256)
                self.activation_layer = torch.nn.ReLU
                
            elif 'bert' in model_name:
                # model_name = 'bert-base-uncased'
                assert self.num_classes is not None
                assert self.task is not None
                
                config_class = BertConfig
                model_class = BertForSequenceClassification
                
                self.config = config_class.from_pretrained(model_name,
                                                           num_labels=self.num_classes,
                                                           finetuning_task=self.task)
                model = model_class.from_pretrained(model_name,
                                                    from_tf=False,
                                                    config=self.config)
                self.activation_layer = 'backbone.bert.pooler.activation'
                
            if self.checkpoint is not None:
                try:
                    state_dict = self.checkpoint['model_state_dict']
                    for k in list(state_dict.keys()):
                        if k.startswith('fc.') and 'bert' in model_name:  
                            state_dict[f'classifier.{k[3:]}'] = state_dict[k]
                            del state_dict[k]
                    
                    model.load_state_dict(state_dict)
                    print(f'Checkpoint loaded!')
                except Exception as e:
                    print(f'Checkpoint not loaded:')
                    print(f'- {e}')
                
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model
        
        
    def init_projection_head(self, backbone, out_dim, project=True):
        if 'resnet' in self.base_model or 'cnn' in self.base_model or 'mlp' in self.base_model:
            dim_mlp = backbone.fc.in_features
            
            self.classifier = nn.Linear(dim_mlp, self.num_classes)
            if project:
                # Modify classifier head to match projection output dimension
                backbone.fc = nn.Linear(dim_mlp, out_dim)
                # Add projection head
                backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), 
                                            nn.ReLU(), 
                                            backbone.fc)
            else:
                backbone.fc = nn.Identity(dim_mlp, -1)
            
        elif 'bert' in self.base_model:
            print(backbone)
            dim_mlp = backbone.classifier.in_features
            
            self.classifier = deepcopy(backbone.classifier)
            print(self.classifier)
            if project:
                backbone.classifier = nn.Linear(dim_mlp, out_dim)
                backbone.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                    nn.ReLU(),
                                                    backbone.classifier)
            else:
                backbone.classifier = nn.Identity(dim_mlp, -1)
                print(backbone.classifier)
        self.dim_mlp = dim_mlp
        return backbone
    
    def forward(self, x):
        if self.base_model == 'bert-base-uncased':
            input_ids, input_masks, segment_ids, labels = x
            outputs = self.backbone(input_ids=input_ids,
                                    attention_mask=input_masks,
                                    token_type_ids=segment_ids,
                                    labels=labels)
            if labels is None:
                return outputs.logits
            return outputs[1]  # [1] returns logits
        return self.backbone(x)
    
    def encode(self, x):
        if self.base_model == 'bert-base-uncased':
            input_ids   = x[:, :, 0]
            input_masks = x[:, :, 1]
            segment_ids = x[:, :, 2]
            x = (input_ids, input_masks, segment_ids, None)
        
        if self.projection_head:
            encoder = deepcopy(self.backbone)
            encoder.fc = nn.Identity(self.dim_mlp, -1)
            if self.base_model == 'bert-base-uncased':
                input_ids, input_masks, segment_ids, labels = x
                return encoder(input_ids=input_ids,
                               attention_mask=input_masks,
                               token_type_ids=segment_ids,
                               labels=labels)
            return encoder(x)
        else:
            return self.forward(x)
    

"""
DevilNetLoss — Devil-NET contrastive loss
==========================================

Mirrors the structure of SupervisedContrastiveLoss but replaces
the single-model contrastive batch with Angel/Devil dual signals:

  Original CnC:
    sim(f(anchor), f(positive))   ← both run through same model
    sim(f(anchor), f(negative))

  Devil-NET:
    sim(z, z_angel_j)  for positives  ← z = f_enc(anchor), z_angel = Angel_enc(x_j)
    sim(z, z_devil_j)  for negatives  ← z_devil = Devil_enc(x_j)

  Self signal (Table 1):
    sim(z, z_angel_self)  ← pull anchor toward its own Angel representation
    sim(z, z_devil_self)  ← push anchor away from its own Devil representation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode


DEFAULT_WEIGHTS = {
    # Table 1 — self signal
    'w1': 0.7,   # TT: angel_pos only
    'w2': 1.0,   # TF: angel_pos component
    'w3': 1.0,   # TF: devil_neg component
    'w4': 0.5,   # FF: devil_neg
    'w5': 0.1,   # FT: devil_neg (conservative)
    # Table 2 — batch positives
    'w_A': 1.0,  # hard  (same class, diff angel pred, diff devil pred)
    'w_B': 0.7,  # soft  (same class, same angel pred, diff devil pred)
    'w_C': 0.3,  # easy  (same class, same angel pred, same devil pred)
    # Table 2 — batch negatives
    'w_E': 1.0,  # hard  (diff class, diff angel pred, same devil pred)
    'w_F': 0.7,  # soft  (diff class, same angel pred, same devil pred)
}


class DevilNetLoss(nn.Module):
    """
    Args:
        signals     : dict output of compute_devil_net_signals(), containing:
                        angel_predictions (N,), devil_predictions (N,),
                        angel_embeddings (N, D), devil_embeddings (N, D), targets (N,)
        weights     : dict of loss weights (see DEFAULT_WEIGHTS)
        temperature : contrastive temperature τ
        alpha       : scale of self loss relative to total
        beta        : scale of batch contrastive loss relative to total
    """

    def __init__(self, signals: dict, weights: dict = None,
                 temperature: float = 0.07, alpha: float = 1.0, beta: float = 1.0,
                 hard_neg_factor: float = 0.0):
        """
        hard_neg_factor : if > 0, dynamically upweights negatives the model already
                          finds similar to the anchor (most confused pairs contribute more).
                          Set to 0 to disable (default). Mirrors CnC's hard_negative_factor.
        """
        super().__init__()
        self.tau             = temperature
        self.alpha           = alpha
        self.beta            = beta
        self.hard_neg_factor = hard_neg_factor
        self.w               = DEFAULT_WEIGHTS.copy()
        if weights:
            self.w.update(weights)

        self.sim = nn.CosineSimilarity(dim=1)

        # Store signals — moved to device on first forward call
        self.angel_pred = torch.from_numpy(signals['angel_predictions'].astype(np.int64))
        self.devil_pred = torch.from_numpy(signals['devil_predictions'].astype(np.int64))
        self.targets    = torch.from_numpy(signals['targets'].astype(np.int64))

        # Pre-normalize embeddings once — shape (N, D)
        self.angel_emb = F.normalize(
            torch.from_numpy(signals['angel_embeddings'].astype(np.float32)), dim=1)
        self.devil_emb = F.normalize(
            torch.from_numpy(signals['devil_embeddings'].astype(np.float32)), dim=1)

        self._on_device = False

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _to_device(self, device):
        if not self._on_device:
            self.angel_pred = self.angel_pred.to(device)
            self.devil_pred = self.devil_pred.to(device)
            self.targets    = self.targets.to(device)
            self.angel_emb  = self.angel_emb.to(device)
            self.devil_emb  = self.devil_emb.to(device)
            self._on_device = True

    # ------------------------------------------------------------------
    # Table 1 — self loss
    # Mirrors compute_exp_sim but for self-alignment:
    #   angel_pos: pull z toward z_angel of the same sample
    #   devil_neg: push z away from z_devil of the same sample
    # ------------------------------------------------------------------

    def compute_self_loss(self, idx: torch.Tensor, z: torch.Tensor):
        """
        Args:
            idx : (B,) dataset indices for this batch
            z   : (B, D) L2-normalized f_enc embeddings

        Returns scalar self loss.
        """
        w = self.w

        # Classify each sample by Table 1 case
        angel_correct = self.angel_pred[idx] == self.targets[idx]   # (B,)
        devil_correct = self.devil_pred[idx]  == self.targets[idx]   # (B,)

        TT =  angel_correct &  devil_correct
        TF =  angel_correct & ~devil_correct
        FF = ~angel_correct & ~devil_correct
        FT = ~angel_correct &  devil_correct

        # cos sim between anchor z and its own angel/devil embeddings
        angel_sim = self.sim(z, self.angel_emb[idx])   # (B,)
        devil_sim  = self.sim(z, self.devil_emb[idx])  # (B,)

        loss = torch.zeros(len(idx), device=z.device)
        loss[TT]  = -w['w1'] * angel_sim[TT]
        loss[TF]  = -w['w2'] * angel_sim[TF] + w['w3'] * devil_sim[TF]
        loss[FF]  =  w['w4'] * devil_sim[FF]
        loss[FT]  =  w['w5'] * devil_sim[FT]   # conservative — devil correct but on seen data

        return loss.mean()

    # ------------------------------------------------------------------
    # Table 2 — batch contrastive loss
    # Mirrors forward() of SupervisedContrastiveLoss:
    #   exp_pos[i,j] = exp(sim(z_i, z_angel_j) / τ)  for j in positives of i
    #   exp_neg[i,j] = exp(sim(z_i, z_devil_j) / τ)  for j in negatives of i
    #   loss_i = -log( Σ w_pos·exp_pos / (Σ w_pos·exp_pos + Σ w_neg·exp_neg) )
    # ------------------------------------------------------------------

    def _pair_weights(self, idx: torch.Tensor):
        """
        Build (B, B) positive and negative weight matrices from Table 2 rules.
        Returns pos_w, neg_w — zero means the pair is skipped.
        """
        w = self.w
        y = self.targets[idx]     # (B,)
        a = self.angel_pred[idx]  # (B,)
        d = self.devil_pred[idx]  # (B,)

        same_class = y.unsqueeze(1) == y.unsqueeze(0)   # (B, B)
        same_angel = a.unsqueeze(1) == a.unsqueeze(0)   # (B, B)
        same_devil = d.unsqueeze(1) == d.unsqueeze(0)   # (B, B)

        # --- Table 2 positives (same class) ---
        case_A = same_class & ~same_angel & ~same_devil   # hard
        case_B = same_class &  same_angel & ~same_devil   # soft
        case_C = same_class &  same_angel &  same_devil   # easy
        # case_D (same_class & diff_angel & same_devil) → conflict, skip

        pos_w = torch.zeros(len(idx), len(idx), device=y.device)
        pos_w[case_A] = w['w_A']
        pos_w[case_B] = w['w_B']
        pos_w[case_C] = w['w_C']

        # --- Table 2 negatives (different class) ---
        diff_class = ~same_class
        case_E = diff_class & ~same_angel &  same_devil   # hard
        case_F = diff_class &  same_angel &  same_devil   # soft
        # case_G (diff_class & diff_angel & diff_devil) → naturally separated, skip
        # case_H (diff_class & same_angel & diff_devil) → noise, skip

        neg_w = torch.zeros(len(idx), len(idx), device=y.device)
        neg_w[case_E] = w['w_E']
        neg_w[case_F] = w['w_F']

        # Exclude self-pairs
        diag = torch.eye(len(idx), dtype=torch.bool, device=y.device)
        pos_w[diag] = 0.
        neg_w[diag] = 0.

        return pos_w, neg_w

    def compute_batch_loss(self, idx: torch.Tensor, z: torch.Tensor):
        """
        Args:
            idx : (B,) dataset indices
            z   : (B, D) L2-normalized f_enc embeddings

        Returns scalar batch contrastive loss.
        """
        B = len(idx)

        pos_w, neg_w = self._pair_weights(idx)   # (B, B)

        # Similarity matrices — mirrors compute_exp_sim
        #   sim_pos[i, j] = sim(z_i, z_angel_j)  — anchor vs Angel of partner
        #   sim_neg[i, j] = sim(z_i, z_devil_j)  — anchor vs Devil of partner
        z_angel_batch = self.angel_emb[idx]   # (B, D)
        z_devil_batch = self.devil_emb[idx]   # (B, D)

        sim_pos = torch.mm(z, z_angel_batch.t()) / self.tau   # (B, B)
        sim_neg = torch.mm(z, z_devil_batch.t()) / self.tau   # (B, B)

        exp_pos = torch.exp(sim_pos)   # (B, B)
        exp_neg = torch.exp(sim_neg)   # (B, B)

        # Weighted sums per anchor row
        sum_exp_pos = (pos_w * exp_pos).sum(dim=1)   # (B,)

        # Optional dynamic hard negative reweighting (flag: hard_neg_factor > 0)
        # Upweights negatives the model already finds similar — most confused pairs
        # contribute more to the denominator, sharpening the repulsion signal.
        # Mirrors CnC's hard_negative_factor logic: reweight = k * exp_neg / mean(exp_neg)
        if self.hard_neg_factor > 0:
            # Only reweight over actual negative positions (neg_w > 0)
            neg_mask      = neg_w > 0                                      # (B, B)
            mean_exp_neg  = (exp_neg * neg_mask).sum(dim=1, keepdim=True) \
                            / neg_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            reweight      = self.hard_neg_factor * exp_neg / (mean_exp_neg + 1e-8)
            sum_exp_neg   = (neg_w * reweight * exp_neg).sum(dim=1)        # (B,)
        else:
            sum_exp_neg = (neg_w * exp_neg).sum(dim=1)                     # (B,)

        # Only compute loss for anchors that have at least one positive and one negative
        valid = (sum_exp_pos > 0) & (sum_exp_neg > 0)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # -log( sum_exp_pos / (sum_exp_pos + sum_exp_neg) )  — mirrors original log_probs
        log_probs = torch.log(sum_exp_pos[valid]) - \
                    torch.log(sum_exp_pos[valid] + sum_exp_neg[valid])
        loss = -log_probs

        del exp_pos, exp_neg, log_probs
        return loss.mean()

    # ------------------------------------------------------------------
    # Forward — mirrors SupervisedContrastiveLoss.forward()
    # ------------------------------------------------------------------

    def forward(self, idx: torch.Tensor, z_raw: torch.Tensor):
        """
        Args:
            idx   : (B,) LongTensor — dataset indices for this batch
            z_raw : (B, D) FloatTensor — raw f_enc outputs (normalized internally)

        Returns:
            total_loss : scalar — alpha*self_loss + beta*batch_loss
            self_loss  : scalar
            batch_loss : scalar
        """
        self._to_device(z_raw.device)

        z = F.normalize(z_raw, dim=1)   # (B, D) — normalize once, use everywhere

        self_loss  = self.compute_self_loss(idx, z)
        batch_loss = self.compute_batch_loss(idx, z)

        total = self.alpha * self_loss + self.beta * batch_loss
        return total, self_loss, batch_loss
    


    
def compute_outputs(inputs, encoder, classifier, args, 
                    labels=None, compute_loss=False,
                    cross_entropy_loss=None):
    inputs = inputs.to(args.device)
    outputs = encoder.encode(inputs)
    if args.replicate in range(10, 20):
        noise = ((0.01 ** 0.5) * torch.randn(*outputs.shape)).to(args.device)
        outputs = outputs + noise
    
    outputs = classifier(outputs)
    loss = torch.zeros(1)
    
    if compute_loss:
        assert labels is not None; cross_entropy_loss is not None
        labels = labels.to(args.device)
        loss = cross_entropy_loss(outputs, labels)
        if args.arch == 'bert-base-uncased_pt':
            return outputs, loss
        free_gpu([labels], delete=True)
        
    free_gpu([inputs], delete=True)
    return outputs, loss
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
    def forward(self, features):
        """
        Compute loss. 
        Args:
        - features (torch.tensor): Input embeddings, expected in form: 
          [target_feature, positive_feature, negative_features[]]
        Returns:
        - loss (torch.tensor): Scalar loss
        """
        target_features = features[0].repeat(features.shape[0] - 2, 1)
        positive_features = features[1].repeat(features.shape[0] - 2, 1)
        loss = self.triplet_loss(target_features, positive_features, features[2:])
        return loss
        
