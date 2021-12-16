import torch
from torchvision.datasets import MNIST
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

class ContrastiveMNIST(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, augment_transform=None, download=False):
        super(ContrastiveMNIST, self).__init__(root, train, transform, target_transform, download)
        assert augment_transform is not None, 'An augment transformation must be provided'
        self.augment_transform = augment_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        img_transform = self.augment_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # target = nn.functional.one_hot(target, num_classes=10)

        segmentation = (img > 0).int()

        # Coarse segmentation
        # coarse = segmentation.unsqueeze(0).float()
        # coarse = nn.functional.interpolate(coarse, (14, 14))
        # coarse = nn.functional.interpolate(coarse, (28, 28))
        # coarse = coarse.squeeze(0)

        negative_samples = torch.zeros_like(img)

        return {'images': img, 'segmentations': segmentation, 'healthy_images': negative_samples,
                'transformed_images': img_transform, 'labels': target}


if __name__ == '__main__':
    t = transforms.Compose([transforms.ToTensor(), ])
    aug = transforms.Compose([transforms.RandomHorizontalFlip()])
    dataset = ContrastiveMNIST("mnist", train=True, download=True, transform=t, augment_transform=aug)
    a = dataset[0]