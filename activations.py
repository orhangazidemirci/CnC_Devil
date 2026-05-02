"""
Functions to help with feature representations
"""
import numpy as np
import torch
from tqdm import tqdm

from utils import print_header
from utils.visualize import plot_umap
from network import get_output

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        try:
            module_out = module_out.detach().cpu()
            self.outputs.append(module_out)  # .detach().cpu().numpy()
        except Exception as e:
            print(e)
            self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def save_activations(model, dataloader, args):
    """
    Returns:
        embeddings  : np.ndarray (N, D) — mid-layer features
        predictions : np.ndarray (N,)  — predicted class indices
    """
    save_output  = SaveOutput()
    target_layer = dict(model.named_modules()).get(model.activation_layer)

    if target_layer is None:
        raise ValueError(f"Layer '{model.activation_layer}' not found in model.")

    handle = target_layer.register_forward_hook(save_output)

    model.to(args.device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader, desc='Saving activations'):
            inputs  = inputs.to(args.device)
            outputs = get_output(model, inputs, labels, args)
            _, pred = torch.max(outputs, 1)
            predictions.append(pred.cpu())

    embeddings  = np.concatenate([o.detach().cpu().numpy().squeeze()
                                  for o in save_output.outputs])
    predictions = np.concatenate(predictions)

    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(len(embeddings), -1)

    handle.remove()
    save_output.clear()

    return embeddings, predictions


def visualize_activations(net, dataloader, label_types, num_data=None,
                          figsize=(8, 6), save=True, ftype='png',
                          title_suffix=None, save_id_suffix=None, args=None, 
                          cmap='tab10', annotate_points=None,
                          predictions=None, return_embeddings=False):
    """
    Visualize and save model activations

    Args:
    - net (torch.nn.Module): Pytorch neural net model
    - dataloader (torch.utils.data.DataLoader): Pytorch dataloader
    - label_types (str[]): List of label types, e.g. ['target', 'spurious', 'sub_target']
    - num_data (int): Number of datapoints to plot
    - figsize (int()): Tuple of image dimensions, by (height, weight)
    - save (bool): Whether to save the image
    - ftype (str): File format for saving
    - args (argparse): Experiment arguments
    """
    if 'resnet' in args.arch or 'densenet' in args.arch or 'bert' in args.arch or 'cnn' in args.arch or 'mlp' in args.arch:
        total_embeddings, predictions = save_activations(net, dataloader, args)
        print(f'total_embeddings.shape: {total_embeddings.shape}')
        e1 = total_embeddings
        e2 = total_embeddings
        n_mult = 1
    else:
        e1, e2, predictions = save_activations(net, dataloader, args)
        n_mult = 2
    pbar = tqdm(total=n_mult * len(label_types))
    for label_type in label_types:
        # For now just save both classifier ReLU activation layers (for MLP, BaseCNN)
        if save_id_suffix is not None:
            save_id = f'{label_type[0]}{label_type[-1]}_{save_id_suffix}_e1'
        else:
            save_id = f'{label_type[0]}{label_type[-1]}_e1'
#         if title_suffix is not None:
#             save_id += f'-{title_suffix}'
        plot_umap(e1, dataloader.dataset, label_type, num_data, method='umap',
                  offset=0, figsize=figsize, save_id=save_id, save=save,
                  ftype=ftype, title_suffix=title_suffix, args=args,
                  cmap=cmap, annotate_points=annotate_points,
                  predictions=predictions)
        # Add MDS
        plot_umap(e1, dataloader.dataset, label_type, 1000, method='mds',
                  offset=0, figsize=figsize, save_id=save_id, save=save,
                  ftype=ftype, title_suffix=title_suffix, args=args,
                  cmap=cmap, annotate_points=annotate_points,
                  predictions=predictions)
        pbar.update(1)
#         if 'resnet' not in args.arch and 'densenet' not in args.arch and 'bert' not in args.arch:
#             save_id = f'{label_type}_e2'
#             if title_suffix is not None:
#                 save_id += f'-{title_suffix}'
#             plot_umap(e2, dataloader.dataset, label_type, num_data,
#                       offset=0, figsize=figsize, save_id=save_id, save=save,
#                       ftype=ftype, title_suffix=title_suffix, args=args,
#                       cmap=cmap, annotate_points=annotate_points,
#                       predictions=predictions)
#             pbar.update(1)
    if return_embeddings:
        return e1, e2, predictions
    del total_embeddings, predictions
    del e1; e2
    # 


def estimate_y_probs(classifier, attribute, dataloader, 
                     classifier_test_size=0.5, 
                     model=None, embeddings=None, 
                     seed=42, reshape_prior=True, args=None):
    if embeddings is None:
        embeddings, _ = save_activations(model, dataloader, args)
        
    X = embeddings
    y = dataloader.dataset.targets_all[attribute]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=classifier_test_size, random_state=seed)
    
    # Fit linear classifier
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(f'Linear classifier score: {score:<.3f}')
    
    # Compute p(y)
    _, y_prior = np.unique(y_test, return_counts=True)
    y_prior = y_prior / y_prior.sum()
    
    # Compute p(y | X)
    y_post = classifier.predict_proba(X_test)
    
    if reshape_prior:
        y_prior = y_prior.reshape(1, -1).repeat(y_post.shape[0], axis=0)
    return y_post, y_prior


def estimate_mi(classifier, attribute, dataloader,
                classifier_test_size=0.5, 
                model=None, embeddings=None, 
                seed=42, args=None):
    if embeddings is None:
        assert model is not None
        embeddings, _ = save_activations(model, dataloader, args)
    # Compute p(y | x), p(y)
    y_post, y_prior = estimate_y_probs(classifier, attribute, 
                                       dataloader, classifier_test_size, 
                                       model, embeddings, seed, 
                                       reshape_prior=True, args=args)
    min_size = np.min((y_post.shape[-1], y_prior.shape[-1]))
    y_post = y_post[:,:min_size]
    y_prior = y_prior[:,:min_size]
    return np.sum(y_post * (np.log(y_post) - np.log(y_prior)), axis=1).mean()


def compute_activation_mi(attributes, dataloader, 
                          method='logistic_regression',
                          classifier_test_size=0.5, max_iter=1000,
                          model=None, embeddings=None, 
                          seed=42, args=None):
    if embeddings is None:
        assert model is not None
        embeddings, _ = save_activations(model, dataloader, args)
        
    if method == 'logistic_regression':
        clf = LogisticRegression(random_state=seed, max_iter=max_iter)
    else:
        raise NotImplementedError
    
    mi_by_attributes = []
    for attribute in attributes:  # ['target', 'spurious']
        mi = estimate_mi(clf, attribute, dataloader,
                         classifier_test_size, model, embeddings,
                         seed, args)
        mi_by_attributes.append(mi)
    return mi_by_attributes


def compute_align_loss(embeddings, dataloader, measure_by='target', norm=True):
    targets_all = dataloader.dataset.targets_all

    if measure_by == 'target':
        targets_t = targets_all['target']
        targets_s = targets_all['spurious']
    elif measure_by == 'spurious':  # A bit hacky
        targets_t = targets_all['spurious']
        targets_s = targets_all['target']
    
    embeddings_by_class = {}
    for t in np.unique(targets_t):
        tix = np.where(targets_t == t)[0]
        anchor_embeddings = []
        positive_embeddings = []
        for s in np.unique(targets_s):
            six = np.where(targets_s[tix] == s)[0]
            if t == s:  # For waterbirds, colored MNIST only
                anchor_embeddings.append(embeddings[tix][six])
            else:
                positive_embeddings.append(embeddings[tix][six])

        embeddings_by_class[t] = {'anchor': np.concatenate(anchor_embeddings),
                                  'positive': np.concatenate(positive_embeddings)}
        
    all_l2 = []
    for c, embeddings_ in embeddings_by_class.items():  # keys
        embeddings_a = embeddings_['anchor']
        embeddings_p = embeddings_['positive']
        if norm:
            embeddings_a /= np.linalg.norm(embeddings_a)
            embeddings_p /= np.linalg.norm(embeddings_p)

        pairwise_l2 = np.linalg.norm(embeddings_a[:, None, :] - embeddings_p[None, :, :], 
                                     axis=-1) ** 2
        all_l2.append(pairwise_l2.flatten())
        
    return np.concatenate(all_l2).mean()  


def compute_aligned_loss_from_model(model, dataloader, norm, args):
    embeddings, predictions = save_activations(model, dataloader, args)
    return compute_align_loss(embeddings, dataloader, norm)

"""
Legacy
"""
def get_embeddings(net, dataloader, args):
    net.to(args.device)
    test_embeddings = []
    test_embeddings_r = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            embeddings = net.embed(inputs)
            embeddings_r = net.embed(inputs, relu=True)

            test_embeddings.append(embeddings.detach().cpu().numpy())
            test_embeddings_r.append(embeddings_r.detach().cpu().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_embeddings_r = np.concatenate(test_embeddings_r, axis=0)
    return test_embeddings, test_embeddings_r
