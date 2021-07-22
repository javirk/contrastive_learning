import torch
import torchvision
import numpy as np
from sklearn.cluster import KMeans

from utils.logs_utils import write_image_tb
from utils.common_utils import segmentation_to_onehot


def sample_results(model, dataset, num_classes, number_images, device, writer=None, epoch_num=None, debug=False,
                   seed=1234):
    kmeans = KMeans(n_clusters=num_classes)  # kmeans always has num_classes
    if debug:
        num_classes = num_classes + 1
        colors = ['red', 'blue', 'yellow', 'green']  # Everything that is red is considered background by the coarse
        # locator. The rest is from the kmeans
    else:
        colors = ['blue', 'yellow', 'green']
    model.eval()

    rng = np.random.RandomState(seed)
    im_idx = rng.randint(0, len(dataset), number_images)
    o = []
    input_batch = []
    for i in im_idx:
        input_batch.append(dataset[i]['images'].unsqueeze(0).to(device))

    input_batch = torch.cat(input_batch, dim=0)

    pred_batch, kmeans = model.forward_validation(input_batch, kmeans, debug)
    input_batch = ((input_batch + 1) / 2 * 255.).type(torch.uint8)
    pred_batch = segmentation_to_onehot(pred_batch, num_classes)

    pred_batch = pred_batch.cpu()
    input_batch = input_batch.cpu()

    for im, seg in zip(input_batch, pred_batch):
        o.append(torchvision.utils.draw_segmentation_masks(im, seg, colors=colors, alpha=0.3))

    if writer is not None:
        assert epoch_num is not None, 'If a writer is passed, epoch_num is mandatory'
        write_image_tb(writer, o, epoch_num, 'Segmentation')
    else:
        return o, pred_batch, input_batch
