"""
Model architecture
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from collections import OrderedDict
# conda install -c huggingface transformers
from transformers import BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
# from resnet import *


def get_net(args, pretrained=None):
    """
    Return model architecture
    """
    pretrained = args.pretrained if pretrained is None else pretrained
    if args.arch == "base":
        net = BaseNet(input_dim=args.d_causal + args.d_spurious,
                      hidden_dim_1=args.hidden_dim_1)
    elif args.arch == "logistic":
        net = LogisticRegression(input_dim=args.d_causal + args.d_spurious)
    elif 'mlp' in args.arch:
        net = MLP(num_classes=args.num_classes,
                  hidden_dim=args.hidden_dim)
        # net.activation_layer = 'relu'
    elif 'cnn' in args.arch:
        net = CNN(num_classes=args.num_classes)
    elif 'resnet' in args.arch:
        if 'resnet50' in args.arch:
            pretrained = True if '_pt' in args.arch else False
            net = torchvision.models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            d = net.fc.in_features
            net.fc = nn.Linear(d, args.num_classes)
            net.activation_layer = 'avgpool'
        elif 'resnet34' in args.arch:
            pretrained = True if '_pt' in args.arch else False
            net = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            d = net.fc.in_features
            net.fc = nn.Linear(d, args.num_classes)
            net.activation_layer = 'avgpool'
    elif 'densenet' in args.arch:
        pretrained = True if '_pt' in args.arch else False
        net = torchvision.models.densenet121(pretrained=pretrained)
        num_ftrs = net.classifier.in_features
        # add final layer with # outputs in same dimension of labels with sigmoid
        N_LABELS = 2  # originally 14 for pretrained model, but try this
        # activation
        net.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
        net.activation_layer = 'features.norm5'
    elif 'bert' in args.arch:
        if args.arch[-3:] == '_pt':
            model_name = args.arch[:-3]
        else:
            model_name = args.arch
            
        assert args.num_classes is not None
        assert args.task is not None
        
        config_class = BertConfig
        model_class = BertForSequenceClassification
        
        config = config_class.from_pretrained(model_name,
                                              num_labels=args.num_classes,
                                              finetuning_task=args.task)
        net = model_class.from_pretrained(model_name, from_tf=False, 
                                          config=config)
        # Either before or after the nonlinearity
        # net.activation_layer = 'bert.pooler.dense'
        net.activation_layer = 'bert.pooler.activation'
        # print(f'net.activation_layer: {net.activation_layer}')
    else:
        raise NotImplementedError
    return net



def load_pretrained_model(path, args):
    checkpoint = torch.load(path)
    net = get_net(args)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    return net


def save_checkpoint(model, optim, loss, epoch, batch, args,
                    replace=True, retrain_epoch=None):
    optim_state_dict = optim.state_dict() if optim is not None else None
    save_dict = {'epoch': epoch,
                 'batch': batch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optim_state_dict,
                 'loss': loss}
    if retrain_epoch is not None:
        epoch = f'{epoch}-cpre={retrain_epoch}'
    cpb_str = f'-cpb={batch}' if batch is not None else ''
    fname = f'cp-{args.experiment_name}-cpe={epoch}{cpb_str}.pt'  # h.tar'
    
    fpath = os.path.join(args.model_path, fname)
    
    # Create directory if it doesn't exist and handle long paths
    fpath = os.path.abspath(fpath)
    if len(fpath) > 260:
        fpath = '\\\\?\\' + fpath
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    
    print(f'replace: {replace}')
    if replace is True:
        model_dir = os.path.dirname(fpath)
        for f in os.listdir(model_dir):
            if f.split('-cpe=')[0] == fname.split('-cpe=')[0]:
                # This one may not be necessary
                if (f.split('-cpe=')[-1].split('-')[0] != str(epoch) or 
                    f.split('-cpb=')[-1].split('.')[0] != str(batch)):
                    print(f'-> Updating checkpoint {f}...')
                    os.remove(os.path.join(model_dir, f))
    if args.dataset == 'isic':
        fpaths = fpath.split('-r=210')
        fpath = fpaths[0] + fpaths[-1]
    try:
        torch.save(save_dict, fpath)
        print(f'Checkpoint saved at {fpath}')
    except Exception as e:
        print(f'Failed to save at {fpath}, error: {e}')
        # Fallback: save in current directory
        torch.save(save_dict, fname)
        print(f'Checkpoint saved at {fname}')
    del save_dict
    return fname

import torch.optim as optim


def get_optim(net, args, model_type='main', scheduler_lr=None):
    """
    Build optimizer for a given model type.

    model_type:
        'main'       — main Devil-NET encoder+classifier
                       uses two parameter groups: encoder (weight_decay)
                       vs classifier (weight_decay_c)
        'spurious'   — stage 1 bias model
                       uses lr_s / momentum_s / weight_decay_s
        'classifier' — standalone linear classifier head only
                       uses lr / momentum / weight_decay_c

    Args:
        net          : nn.Module
        args         : args namespace
        model_type   : str (see above)
        scheduler_lr : float or None — overrides lr when provided (for LR schedulers)

    Returns:
        optimizer : torch.optim.Optimizer
    """

    # ------------------------------------------------------------------
    # Select hyperparameters by model type
    # ------------------------------------------------------------------
    if model_type == 'spurious':
        lr           = args.lr_s
        momentum     = args.momentum_s
        weight_decay = args.weight_decay_s

    elif model_type == 'classifier':
        lr           = args.lr if scheduler_lr is None else scheduler_lr
        momentum     = args.momentum
        weight_decay = args.weight_decay_c

    else:  # 'main'
        lr           = args.lr if scheduler_lr is None else scheduler_lr
        momentum     = args.momentum
        weight_decay = args.weight_decay

    # ------------------------------------------------------------------
    # Build optimizer
    # ------------------------------------------------------------------
    if args.optim == 'sgd':
        if model_type == 'main':
            # Two parameter groups: encoder and classifier regularized separately
            optimizer = optim.SGD(
                _two_group_params(net, args),
                lr=lr,
                momentum=momentum,
            )
        else:
            optimizer = optim.SGD(
                net.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

    elif args.optim == 'adam':
        if model_type == 'main':
            optimizer = optim.Adam(
                _two_group_params(net, args),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        else:
            optimizer = optim.Adam(
                net.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )

    elif args.optim == 'AdamW':
        # AdamW: no weight decay on bias and LayerNorm (standard for BERT)
        no_decay = ['bias', 'LayerNorm.weight']

        if model_type == 'main':
            # Four groups: (encoder / classifier) × (decay / no-decay)
            optimizer_params = [
                {
                    'params': [p for n, p in net.named_parameters()
                               if _is_encoder(n, net)
                               and not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay,
                    'lr': lr,
                },
                {
                    'params': [p for n, p in net.named_parameters()
                               if _is_encoder(n, net)
                               and any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                },
                {
                    'params': [p for n, p in net.named_parameters()
                               if not _is_encoder(n, net)
                               and not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay_c,
                    'lr': lr,
                },
                {
                    'params': [p for n, p in net.named_parameters()
                               if not _is_encoder(n, net)
                               and any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                },
            ]
            # Drop empty groups
            optimizer_params = [g for g in optimizer_params if g['params']]
        else:
            optimizer_params = [
                {
                    'params': [p for n, p in net.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    'weight_decay': weight_decay,
                },
                {
                    'params': [p for n, p in net.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
            ]

        optimizer = optim.AdamW(optimizer_params, lr=lr, eps=1e-8)

    else:
        raise NotImplementedError(f"Optimizer '{args.optim}' not supported. "
                                  f"Choose from: sgd, adam, AdamW")

    return optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_group_params(net, args):
    """
    Split net parameters into two groups:
      - encoder parameters : weight_decay
      - classifier parameters : weight_decay_c
    """
    encoder_params    = []
    classifier_params = []

    for name, param in net.named_parameters():
        if _is_encoder(name, net):
            encoder_params.append(param)
        else:
            classifier_params.append(param)

    groups = []
    if encoder_params:
        groups.append({'params': encoder_params,    'weight_decay': args.weight_decay})
    if classifier_params:
        groups.append({'params': classifier_params, 'weight_decay': args.weight_decay_c})

    # Fallback: if split failed (unusual architecture), use all params
    if not groups:
        groups = [{'params': list(net.parameters()), 'weight_decay': args.weight_decay}]

    return groups


def _is_encoder(param_name, net):
    """
    Returns True if param_name belongs to the encoder (not the classifier head).
    Checks common classifier head naming conventions.
    """
    classifier_keywords = ('classifier', 'fc', 'head', 'linear')

    # If the model explicitly declares activation_layer, use that as boundary
    if hasattr(net, 'activation_layer'):
        return not any(kw in param_name for kw in classifier_keywords)

    # Generic fallback: anything named 'classifier', 'fc', 'head', 'linear'
    # at the top level is the classifier
    top_level = param_name.split('.')[0]
    return top_level not in classifier_keywords



class BaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=20):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, 2)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

    def predict(self, x):
        x = F.softmax(self.forward(x))
        return np.argmax(x.detach().numpy())

    def embed(self, x, relu=False):
        x = self.fc1(x)
        return F.relu(self.fc2(x)) if relu else self.fc2(x)

    def last_layer_output(self, x, relu=False):
        """
        Opposite to embed, return softmax based on last layer
        Args:
        - x (torch.tensor): neural network embeddings
        - relu (bool): Whether x has ReLU applied to it
        Output:
        - Neural network output given hidden layer representation
        """
        return self.fc(x) if relu else self.fc(F.relu(x))


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc1(x)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5
        self.fc2 = nn.Linear(120, 84)  # Activations layer
        self.fc = nn.Linear(84, num_classes)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        
        self.activation_layer = torch.nn.ReLU

    def forward(self, x):
        # Doing this way because only want to save activations
        # for fc linear layers - see later
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x))
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        
        self.activation_layer = torch.nn.ReLU

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.relu_1(self.fc1(x))
        x = self.relu_2(self.fc2(x))
        x = self.fc(x)
        return x
