"""
Training, evaluating, calculating embeddings functions
"""
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import get_optim
from network import save_checkpoint
from utils import print_header
from utils.logging import summarize_acc
from utils.metrics import compute_roc_auc
from activations import compute_activation_mi, save_activations, compute_align_loss

