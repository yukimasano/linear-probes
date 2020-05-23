import os
import glob
import torch


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    if path is not None and not os.path.exists(path):
        os.makedirs(path)


def get_model_device(model):
    return next(model.parameters()).device


def load_model(model, model_path):
    """Load the model state from the model_path PTH file. The latter can also be a dictionary with a `model` key, which is then used as model state dictionary instead.
    If model_path is None it does not do anything.
    """
    if model_path is None:
        return
    data = torch.load(model_path, map_location=str(get_model_device(model)))
    if 'model' in data:
        data = data['model']
    elif 'state_dict' in data:
        # deep cluster case
        model.top_layer = None
        data = (data['state_dict'])
        for key in list(data.keys()):
            if 'top_layer' in key:
                del data[key]
        for k in list(data.keys()):
            if "module." in k:
                 data[k.replace('module.', '')] = data.pop(k)  # [k]
            # if "batches" in k:
    model.load_state_dict(data)


def save_model(model, model_path):
    """Save the model state dictionary to the PTH file model_path."""
    if model_path is not None:
        xmkdir(os.path.dirname(model_path))
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, model_path)


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    """Search the latest checkpoint in checkpoint_dir and load the model and optimizer and return the metrics."""
    names = list(sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
    ))
    if len(names) == 0:
        return 0, {'train': [], 'val': []}
    print(f"Loading checkpoint '{names[-1]}'")
    cp = torch.load(names[-1], map_location=str(get_model_device(model)))
    epoch = cp['epoch']
    metrics = cp['metrics']
    if model:
        model.load_state_dict(cp['model'])
    if optimizer:
        optimizer.load_state_dict(cp['optimizer'])
    return epoch, metrics


def clean_checkpoint(checkpoint_dir, dry_run=False):
    names = list(sorted(
        glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
    ))
    if len(names) > 2:
        for name in names[0:-2]:
            if dry_run:
                print(f"Would delete redundant checkpoint file {name}")
            else:
                print(f"Deleting redundant checkpoint file {name}")
                os.remove(name)


def save_checkpoint(checkpoint_dir, model, optimizer, metrics, epoch, defsave=False):
    """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir
    for the specified epoch. If checkpoint_dir is None it does not do anything."""
    if checkpoint_dir is not None:
        xmkdir(checkpoint_dir)
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        if (epoch % 50 == 0) or defsave:
            name = os.path.join(checkpoint_dir, f'ckpt{epoch:04}.pth')
        else:
            name = os.path.join(checkpoint_dir, f'checkpoint{epoch:04}.pth')
        torch.save({
            'epoch': epoch + 1,
            'metrics': metrics,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
        }, name)
        clean_checkpoint(checkpoint_dir)









