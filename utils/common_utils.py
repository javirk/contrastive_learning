import yaml
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import torchvision.transforms as transforms
import math
import numpy as np
import sys
from data.data_retriever import SegmentationDataset, ContrastiveDataset


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def str2bool(value, raise_exc=False):
    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}
    if isinstance(value, str) or sys.version_info[0] < 3:
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def prepare_run(root_path, config_path):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = root_path.joinpath('runs/TL_{}'.format(current_time))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)
    copy_file(config_path, f'{tb_path}/config.yml')
    return writer, device, current_time


def get_train_transformations(s=1):
    augmentation = [  # transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        transforms.RandomAffine(25, translate=(0.25, 0.25), scale=(0.8, 1.2), fill=-1),
        # -1 because they are normalized (-1,1)
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 3))
    ]

    return transforms.Compose(augmentation)


def get_val_transformations():
    augmentation = []
    return transforms.Compose(augmentation)


def get_dataset(p, data_path, mode, common_transform=None, augment_transform=None):
    dataset_name = p[f'{mode}_kwargs']['dataset'].lower()
    if dataset_name == 'ambulatorium':
        return ContrastiveDataset(data_path.joinpath('ambulatorium_all.hdf5'), common_transform=common_transform,
                                  augment_transform=augment_transform, n_classes=p['num_classes'])
    elif dataset_name == 'oct_test':
        return ContrastiveDataset(data_path.joinpath('oct_test_all.hdf5'), common_transform=common_transform,
                                  augment_transform=augment_transform, n_classes=p['num_classes'])
    elif dataset_name == 'retouch':
        volume_path = data_path.joinpath('Segmentation/RETOUCH/Spectralis_volume.npy')
        labels_path = data_path.joinpath('Segmentation/RETOUCH/Spectralis_labels.npy')
        return SegmentationDataset(volume_path, labels_path, transform=common_transform, only_fluid=True)
    else:
        raise ValueError(f'Invalid dataset {dataset_name}')


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    elif p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def segmentation_to_onehot(seg, num_classes):
    """
    Turns a segmentation in the format NxHxW to NxCxHxW (one hot)
    :param seg: torch.Tensor. Shape NxHxW
    :param num_classes: int. Number of classes
    :return: Boolean torch.Tensor with shape NxCxHxW. It's boolean because torchvision.utils.draw_segmentation_masks
    wants it that way
    """
    seg = seg.long()
    if len(seg.shape) == 3:
        seg = seg.unsqueeze(1)
    one_hot = torch.FloatTensor(seg.size(0), num_classes, seg.size(2), seg.size(3)).zero_()
    one_hot.scatter_(1, seg, 1)
    return one_hot.bool()


def otsu_thresholding(image_batch, prototypes):
    import numpy as np
    from skimage.filters import threshold_otsu
    from skimage.morphology import opening, closing
    from einops import rearrange

    batch_size, _, h, w = image_batch.size()
    image_batch = image_batch.cpu()

    binary_im = torch.zeros((batch_size, h, w), device=prototypes.device)
    for i in range(batch_size):
        im_eroded = opening(image_batch[i,0])
        im_closed = closing(im_eroded, np.ones((5, 5)))
        binary_im[i] = torch.from_numpy(im_closed > threshold_otsu(im_closed)).int()

    binary_im = rearrange(binary_im, 'b h w -> b (h w)')
    prototypes = rearrange(prototypes, 'b d h w -> b d (h w)')
    prototypes = torch.bmm(prototypes, binary_im.unsqueeze(-1)).squeeze(-1)  # N x dim
    return prototypes