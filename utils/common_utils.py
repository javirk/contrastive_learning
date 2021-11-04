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
import data.transforms_segmentation as t


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
    augmentation = []

    return transforms.Compose(augmentation)


def get_val_transformations():
    augmentation = [t.CLAHE(), t.ToTensor(), t.Resize(512), t.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    return t.Compose(augmentation)


def get_paths_validation(p, data_path):
    if p['val_kwargs']['dataset'] == 'retouch':
        seg_path = data_path.joinpath('Segmentation', 'RETOUCH')
        volumes_path = seg_path.joinpath('Spectralis_volume.npy')
        labels_path = seg_path.joinpath('Spectralis_labels.npy')
    elif p['val_kwargs']['dataset'] == 'oct_test':
        seg_path = data_path.joinpath('Segmentation', 'retinai')
        volumes_path = seg_path.joinpath('volume.npy')
        labels_path = seg_path.joinpath('segmentation.npy')
    else:
        raise ValueError(f'{p["val_kwargs"]["dataset"]} dataset not understood')

    return volumes_path, labels_path


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
        im_eroded = opening(image_batch[i, 0])
        im_closed = closing(im_eroded, np.ones((5, 5)))
        binary_im[i] = torch.from_numpy(im_closed > threshold_otsu(im_closed)).int()

    binary_im = rearrange(binary_im, 'b h w -> b (h w)')
    prototypes = rearrange(prototypes, 'b d h w -> b d (h w)')
    prototypes = torch.bmm(prototypes, binary_im.unsqueeze(-1)).squeeze(-1)  # N x dim
    return prototypes


def calculate_IoU(preds, labels, threshold=0.5, reduction='mean'):
    if type(preds) == torch.Tensor:
        preds = (preds > threshold).int()
        labels = labels.int()
        # Taken from: https://discuss.pytorch.org/t/understanding-different-metrics-implementations-iou/85817
        intersection = (preds & labels).float().sum(dim=(-2, -1))  # Will be zero if Truth=0 or Prediction=0
        union = torch.clip((preds | labels).float().sum(dim=(-2, -1)), min=1e-10)  # Will be zero if both are 0
        iou = intersection / union

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            iou = torch.mean(iou)
        else:
            raise ValueError('Unknown reduction type')
    else:
        preds = (preds > threshold).astype(int)
        labels = (labels > threshold).astype(int)
        intersection = np.sum((preds & labels), axis=(-2, -1))  # Will be zero if Truth=0 or Prediction=0
        union = np.sum((preds | labels), axis=(-2, -1))  # Will be zero if both are 0

        iou = intersection / union

        if reduction == 'none':
            pass
        elif reduction == 'mean':
            iou = np.nanmean(iou)
        else:
            raise ValueError('Unknown reduction type')

    return iou


def IoU_per_class(pred, labels, num_classes, threshold=0.5):
    '''

    :param pred: torch.Tensor([B, H, W], dtype=torch.int32). 0 in the coarse background, 1 and 2 for the cluster classes
    This means that we have to compare 1 and 2 to the labels, which are 0 and 1. We substract 1 to preds and the rest is
    ok (but now background is -1)
    :param labels:
    :param num_classes:
    :param threshold:
    :return:
    '''
    pred = pred - 1
    iou = torch.zeros([pred.shape[0], num_classes, num_classes], dtype=torch.float)
    for i_pred in range(num_classes):
        pred_cls = (pred == i_pred).int()
        for i_lab in range(num_classes):
            label_cls = (labels == i_lab).int()
            iou[:, i_pred, i_lab] = calculate_IoU(pred_cls, label_cls, threshold, reduction='none')

    return iou

def apply_criterion(iou, hungarian, predicted_batch=None):
    bs, _, num_classes = iou.shape
    if predicted_batch is not None:
        pred = predicted_batch.clone()
        pred = pred - 1
    else:
        pred = None
    mean_iou_fluid = 0
    mean_iou_bg = 0

    for i, iou_im in enumerate(iou):
        hungarian.calculate(iou_im, is_profit_matrix=True)  # Profit because higher IoU is better
        mean_iou_fluid += get_potential_fluid(iou_im, hungarian)
        mean_iou_bg += get_potential_bg(iou_im, hungarian)
        res = hungarian.get_results()

        if predicted_batch is not None:
            # Ugly and doesn't work for more than two classes
            pred[i, pred[i] == res[0][1]] = num_classes   # The values to an auxiliary value
            pred[i, pred[i] == res[1][1]] = res[1][0]  # The values to the real label
            pred[i, pred[i] == num_classes] = res[0][0]  # Aux values to the real label

    if predicted_batch is not None:
        pred[pred == -1] = 0
    return pred, mean_iou_fluid / bs, mean_iou_bg / bs

def get_potential_fluid(iou_matrix, hungarian):
    """Fluid is always in the second column (this doesn't work for more than two classes). hungarian.get_results()
    returns the idx of the selected values in Iou_matrix, so we iterate over them to find which one has information
    in the second column."""
    for idx in hungarian.get_results():
        if idx[1] == 1:
            break
    return iou_matrix[idx]

def get_potential_bg(iou_matrix, hungarian):
    """Background is always in the first column (this doesn't work for more than two classes). hungarian.get_results()
    returns the idx of the selected values in Iou_matrix, so we iterate over them to find which one has information
    in the second column."""
    for idx in hungarian.get_results():
        if idx[0] == 1:
            break
    return iou_matrix[idx]