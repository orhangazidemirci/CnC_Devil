"""
Logging functions and classes
"""
import os
import sys
import csv
import numpy as np

def log_epoch(epoch, train_losses, val_results):
    line = (f'Epoch {epoch:3d} | '
            f'loss={train_losses["total"]:.4f}  '
            f'ce={train_losses["ce"]:.4f}  '
            f'self={train_losses["self"]:.4f}  '
            f'batch={train_losses["batch"]:.4f} | '
            f'val_avg={val_results["avg_acc"]*100:.1f}%')
    if 'worst_group' in val_results:
        line += f'  val_worst={val_results["worst_group"]*100:.1f}%'
    print(line)


def log_devil_signals(sliced_outputs):
    targets = sliced_outputs['targets']
    a_pred  = sliced_outputs['angel_predictions']
    d_pred  = sliced_outputs['devil_predictions']
    N       = len(targets)

    angel_correct = (a_pred == targets).mean() * 100
    devil_correct = (d_pred == targets).mean() * 100
    tf_case = ((a_pred == targets) & (d_pred != targets)).mean() * 100
    tt_case = ((a_pred == targets) & (d_pred == targets)).mean() * 100
    ff_case = ((a_pred != targets) & (d_pred != targets)).mean() * 100
    ft_case = ((a_pred != targets) & (d_pred == targets)).mean() * 100

    print(f'\n--- Devil-NET Signal Quality (N={N}) ---')
    print(f'  Angel correct:               {angel_correct:.1f}%')
    print(f'  Devil correct:               {devil_correct:.1f}%')
    print(f'  TF — strongest signal:       {tf_case:.1f}%')
    print(f'  TT — angel pos only:         {tt_case:.1f}%')
    print(f'  FF — devil neg only:         {ff_case:.1f}%')
    print(f'  FT — conservative neg:       {ft_case:.1f}%')
    print('-' * 42)


def log_partition_bias(train_loaders, group_col=0):
    print('\n--- Partition Bias Summary ---')
    for i, loader in enumerate(train_loaders):
        dataset = loader.dataset
        if hasattr(dataset, 'indices') and hasattr(dataset.dataset, 'metadata_array'):
            meta   = dataset.dataset.metadata_array[dataset.indices, group_col]
            groups = meta.numpy() if hasattr(meta, 'numpy') else np.array(meta)
            n_spur = (groups % 2 == 0).sum()
            ratio  = n_spur / len(groups)
            print(f'  Partition {i}: {len(groups):5d} samples  '
                  f'spurious ratio={ratio:.3f}')
        else:
            print(f'  Partition {i}: {len(loader.dataset):5d} samples  '
                  f'(no metadata available)')
    print('-' * 40)

def summarize_acc(correct_by_groups, total_by_groups, stdout=True):
    all_correct = 0
    all_total = 0
    min_acc = 101.
    min_correct_total = [None, None]
    if stdout:
        print('Accuracies by groups:')
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            if acc < min_acc:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
        robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
        print('-' * len(average_str))
        print(average_str)
        print(robust_str)
        print('-' * len(average_str))
    return all_correct / all_total * 100, min_acc


def initialize_csv_metrics(args):
    test_metrics = {
        'epoch':          [],
        'target':         [],
        'spurious':       [],
        'acc':            [],
        'loss':           [],
        'loss_ce':        [],   # new
        'loss_self':      [],   # new
        'loss_batch':     [],   # new
        'model_type':     [],
        'robust_acc':     [],
        'max_robust_acc': [],
    }
    args.test_metrics    = test_metrics
    args.max_robust_acc  = 0
    args.max_robust_epoch = 0   # new — track which epoch was best


class Logger(object):
    """
    Print messages to stdout and save to specified filed

    Args:
    - fpath (str): Destination path for saving logs
    - mode (str): How to edit the opened file at fpath (default 'w')
    """

    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        
        if fpath is not None:
            # Normalize path to absolute path
            fpath = os.path.abspath(fpath)
            
            # Handle Windows long path limit (260 chars)
            if os.name == 'nt' and len(fpath) > 260:
                fpath = '\\\\?\\' + fpath
            
            dir_path = os.path.dirname(fpath)
            
            # Create ALL parent directories recursively
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            self.file = open(fpath, mode)
    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg, stdout=True):
        if stdout:
            self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # Never close sys.stdout
        if self.file is not None:
            self.file.close()


def log_args(args, logger):
    """
    Log experimental arguments to logging file
    """
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')
    
    
def log_data(dataset, header, indices=None):
    print(header)
    dataset_groups = dataset.targets_all['group_idx']
    if indices is not None:
        dataset_groups = dataset_groups[indices]
    groups = np.unique(dataset_groups)
    
    try:
        max_target_name_len = np.max([len(x) for x in dataset.class_names])
    except:
        max_target_name_len = -1
    
    for group_idx in groups:
        counts = np.where(dataset_groups == group_idx)[0].shape[0]
        try:  # Dumb but arguably more pretty stdout
            group_name = dataset.group_labels[group_idx]
            group_name = group_name.split(',')
            # target_name_len = len(group_name[0]) - max_target_name_len
            group_name[0] += (' ' * int(
                np.max((0, max_target_name_len - len(group_name[0])))
            ))
            group_name = ','.join(group_name)
            print(f'    {group_name} : n = {counts}')
        except Exception as e:
            print(e)
            print(f'    {group_idx} : n = {counts}')
