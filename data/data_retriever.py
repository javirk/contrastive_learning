from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import logging
import random
import skimage.transform
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class SegmentationDataset(Dataset):
    """Segmentation dataset.
        File has 0=Background, 1=IRF, 2=SRF, 3="PED"
    """

    def __init__(self, volume_file, segmentation_file, max_len=None, seed=1234, transform=None, only_fluid=True,
                 no_classes=True, remove_small_segmentations=True):
        self.slices = np.load(volume_file, allow_pickle=True)
        self.segmentation = np.load(segmentation_file, allow_pickle=True)

        assert self.slices.shape == self.segmentation.shape, 'Shapes of segmentation and volumes must be equal'

        self.transform = transform

        # This is to have the same labels as in the OCTHDF5Dataset dataset.
        self.segmentation = np.where(self.segmentation == 3, 0, self.segmentation)  # Remove PED
        # self.segmentation = np.where(self.segmentation == 2, 3, self.segmentation)  # AUX
        # self.segmentation = np.where(self.segmentation == 1, 2, self.segmentation)  # IRF to index 2
        # self.segmentation = np.where(self.segmentation == 3, 1, self.segmentation)  # SRF to index 1

        if no_classes:
            # Do not distinguish IRF and SRF
            self.segmentation = self.segmentation.astype(bool).astype(int)
            self.n_classes = 2
        else:
            self.n_classes = 3

        if only_fluid:
            idx_keep = (self.segmentation != 0).any(axis=(2, 3)).astype(float).nonzero()[0]
            self.slices = self.slices[idx_keep]
            self.segmentation = self.segmentation[idx_keep]

        if remove_small_segmentations:
            # 50 is the size of the mask. Any segmentation under 50 pixels (total per image) is discarded
            idx_keep = (self.segmentation.sum(axis=(2,3)) > 50).astype(float).nonzero()[0]
            self.slices = self.slices[idx_keep]
            self.segmentation = self.segmentation[idx_keep]

        if max_len is not None and max_len != -1:
            np.random.seed(seed)
            shuffler = np.random.permutation(len(self.slices))
            self.slices = self.slices[shuffler][:max_len]
            self.segmentation = self.segmentation[shuffler][:max_len]

        self.weights = self._get_weights()
        self.dataset_len = self.segmentation.shape[0]

    def __len__(self):
        return self.dataset_len

    def _get_weights(self):
        n_samples = []
        for i in range(self.n_classes):
            n_samples.append(np.sum(self.segmentation == i))

        normed_weights = [1 - (x / np.sum(n_samples)) for x in n_samples]
        normed_weights = torch.as_tensor(normed_weights, dtype=torch.float)

        return normed_weights

    def __getitem__(self, idx):
        image = self.slices[idx][0] / 256
        segmentation = self.segmentation[idx][0]

        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        if self.transform is not None:
            image, segmentation = self.transform(image, segmentation)

        # segmentation = segmentation.squeeze().long()

        return {'images': image, 'segmentations': segmentation}


class ContrastiveDataset(Dataset):
    """OCT HDF5 dataset."""

    def __init__(self, hdf5_file, image_set="data/slices", label_set="data/markers", common_transform=None,
                 augment_transform=None, n_classes=10):
        """
        labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
        """
        assert augment_transform is not None, 'Augmenting transform cannot be None'
        self.hdf5_file = hdf5_file
        self.image_set_name = image_set
        self.label_set_name = label_set

        self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]
        self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]

        self.common_transform = common_transform
        self.augment_transform = augment_transform
        if n_classes != 10:
            self._remove_unused_labels()
        self.healthy_idx = np.where(self.label_set[:,0] == 1)[0]
        self.nonhealthy_idx = np.where(self.label_set[:,0] != 1)[0]

        # self.label_set_disease = self.label_set[nonhealthy_idx]
        self.dataset_len = len(self.nonhealthy_idx)
        self.healthy_len = len(self.healthy_idx)


    def __len__(self):
        return self.dataset_len

    def _remove_unused_labels(self):
        self.label_set = self.label_set[:, :3]  # Only until IRF
        self.label_set[:, 0] = np.logical_not(np.logical_or(self.label_set[:, 1], self.label_set[:, 2])).astype(int)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_set[self.nonhealthy_idx[idx]]
        image_healthy = self.image_set[self.healthy_idx[np.random.randint(0, self.healthy_len)]]
        label = self.label_set[self.nonhealthy_idx[idx]].astype(np.float32)

        image = np.concatenate([image, image, image], axis=-1)
        image_healthy = np.concatenate([image_healthy, image_healthy, image_healthy], axis=-1)

        seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.common_transform:
            random.seed(seed)
            image = self.common_transform(image)
            image_healthy = self.common_transform(image_healthy)

        image_transform = self.augment_transform(image)

        sample = {'images': image, 'healthy_images': image_healthy, 'transformed_images': image_transform, 'labels': label}
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t_seg = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    t_aug = transforms.Compose([
        transforms.RandomAffine(20, translate=(0.15, 0.15), scale=(0.9, 1.1), fill=-1),
        transforms.RandomHorizontalFlip()])

    d = ContrastiveDataset('../../../Datasets/ambulatorium_all.hdf5', common_transform=t_seg, augment_transform=t_aug)

    im = d[0]