import yaml
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
from utils.logs_utils import write_image_tb
from sklearn.cluster import KMeans


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def prepare_run(root_path, config_path):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = root_path.joinpath('runs/TL_{}'.format(current_time))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)
    copy_file(config_path, f'{tb_path}/config.yml')
    return writer, device, current_time


def get_train_transformations(s=1):
    augmentation = [#transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                    transforms.RandomAffine(20, translate=(0.25, 0.25), scale=(0.8, 1.2), fill=-1),
                    # -1 because they are normalized (-1,1)
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                    ]

    return transforms.Compose(augmentation)


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


def sample_results(model, dataset, num_classes, writer, epoch_num, number_images, device):
    kmeans = KMeans(n_clusters=num_classes)
    model.eval()
    im_idx = np.random.randint(0, len(dataset), number_images)
    o = []
    for i in im_idx:
        input_batch = dataset[i]['images'].unsqueeze(0).to(device)

        pred_batch, kmeans = model.forward_validation(input_batch, kmeans)
        input_batch = ((input_batch + 1) / 2 * 255.).type(torch.uint8)
        pred_batch = segmentation_to_onehot(pred_batch, num_classes)

        input_batch = input_batch.to(pred_batch.device)

        o.append(torchvision.utils.draw_segmentation_masks(input_batch[0], pred_batch[0]))

    write_image_tb(writer, o, epoch_num, 'Segmentation')
