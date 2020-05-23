import torchvision
import torch
import torchvision.transforms as tfs
import os



def get_standard_data_loader(image_dir, is_validation=False,
    batch_size=256, image_size=256, crop_size=224,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], num_workers=8,
    no_random_crops=False, tencrops=True):
    """Get a standard data loader that can be used for ImageNet and similar datasets.

    By default image are color normalized, randomly resized and
    cropped in a 224 x 224 pixels square and randomly flipped
    left-to-right. Images are also scanned in random order (shuffled).
    If is_validation is True, this is changed so that the crop is
    centered after resizing images t 256 pixels and no left-to-right
    flipping nor shuffling is applied.

    Keyword Arguments:
        image_dir : string
            Directory of subdirectory of images, one per class. Set to Null to
            return Null as data loader. The image directory has format:
                <image_dir>/class1/file001.jpg
                <image_dir>/class1/file002.jpg
                ...
                <image_dir>/class2/file001.jpg
                ...
        batch_size : int
            Size of a batch of data in number of images (default: 256)
        is_validation : bool
            Set to True to create a data loader for validation (default: False)
        image_size : int
            Size of the image before cropping (default: 256)
        crop_size : int
            Size of the image crop (default: 224)
        mean : [float]
            Mean for RGB normalization (default: [0.485, 0.456, 0.406])
        std : [float]
            Standard deviation for RGB normalization (default: [0.229, 0.224, 0.225])
        num_workers : int
            Number of loader threads. Set to 0 to load from the main thread (default: 0
        no_random_crops : False
            Set to True to avoid random crops augmentation for training (but still use horizontal flips).
    Returns:
        loader : DataLoader
            A DataLoader object.
    """
    if image_dir is None:
        return None
    normalize = tfs.Normalize(mean=mean, std=std)
    if is_validation:
        if tencrops:
             transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.TenCrop(crop_size),
                tfs.Lambda(lambda crops: torch.stack([normalize(tfs.ToTensor()(crop)) for crop in crops]))
            ])
             batch_size = int(batch_size/10)
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.ToTensor(),
                normalize
            ])
    else:
        if not no_random_crops:
            transforms = tfs.Compose([
                tfs.RandomResizedCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])

    dataset = torchvision.datasets.ImageFolder(image_dir, transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None
    )
    return loader

def get_standard_data_loader_pairs(dir_path, **kargs):
    """Get a pair of data loaders for training and validation."""
    train = get_standard_data_loader(os.path.join(dir_path, "train"), is_validation=False, **kargs)
    val = get_standard_data_loader(os.path.join(dir_path, "val"), is_validation=True, **kargs)
    return train, val