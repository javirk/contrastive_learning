import torch
import torchvision
import numpy as np
import os
from einops import rearrange
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

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

    pred_batch, kmeans = model.module.forward_validation(input_batch, kmeans, debug)
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


@torch.no_grad()
def save_embeddings_to_disk(p, val_loader, model, seed=1234, device='cpu'):
    """
    Inspired from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation
    :param p:
    :param val_loader:
    :param model:
    :param seed:
    :param device:
    :return:
    """
    import torch.nn as nn
    print('Save kmeans embeddings to disk ...')
    model.eval()
    ptr = 0
    if p['val_kwargs']['coarse_pixels_only']:
        filename = os.path.join(p['embedding_dir'], 'embeddings_kmeans_coarseonly.npy')
    else:
        filename = os.path.join(p['embedding_dir'], 'embeddings_kmeans.npy')

    all_embeddings = torch.zeros((len(val_loader.sampler), 496, 512)).to(device)
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']
        cls = qdict['cls'].sigmoid().cpu()

        b, c, h, w = features.shape
        features = rearrange(features, 'b dim h w -> (b h w) dim')  # features: pixels x dim

        if p['val_kwargs']['coarse_pixels_only']:
            coarse = qdict['cls_emb']
            coarse = (torch.softmax(coarse, dim=1) > p['model_kwargs']['coarse_threshold'])
            coarse = coarse.int().argmax(dim=1)  # Prediction of each pixel (coarse). B x H x W
            coarse = (coarse != 0).reshape(-1)  # True/False. B.H.W (pixels)
            coarse_idx = torch.nonzero(coarse).squeeze()

            background_idx = torch.nonzero(coarse == 0).squeeze()
            prototypes_background = torch.index_select(features, index=background_idx, dim=0)  # False pixels x dim
            prototypes_background = nn.functional.normalize(prototypes_background, dim=1)
            mean_background_prototype = torch.mean(prototypes_background, dim=0).cpu()
        else:
            coarse_idx = torch.tensor(range(features.shape[0]), device=features.device)

        prototypes = torch.index_select(features, index=coarse_idx, dim=0)  # True pixels x dim
        prototypes = nn.functional.normalize(prototypes, dim=1)

        n_clusters = (cls > 0.5).sum() + 1  # Detected biomarkers + background
        prototypes = prototypes.cpu()

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)

        if p['val_kwargs']['use_pca']:
            # In the original code they applied PCA before kmeans
            pca = PCA(n_components=32, whiten=True)
            prototypes_pca = pca.fit_transform(prototypes.numpy())
        else:
            prototypes_pca = prototypes.numpy()

        embeddings_kmeans = kmeans.fit_predict(prototypes_pca)

        background_cluster = 0  # This has no effect if coarse_pixels_only is False
        if p['val_kwargs']['coarse_pixels_only']:
            min_dist = 1e8
            for cluster in range(n_clusters):
                embedding_cluster_idx = (embeddings_kmeans == cluster).nonzero()[0]  # They are numpy arrays!
                embedding_cluster = torch.index_select(prototypes, index=torch.tensor(embedding_cluster_idx), dim=0)
                mean_embedding_cluster = torch.mean(embedding_cluster, dim=0)
                dist = torch.dist(mean_embedding_cluster, mean_background_prototype)
                if dist < min_dist:
                    min_dist = dist
                    background_cluster = cluster

        embeddings = torch.zeros((b * h * w), dtype=torch.int32)
        embeddings[coarse_idx] = torch.tensor(embeddings_kmeans) + 1
        embeddings[embeddings == (background_cluster + 1)] = 0
        embeddings = rearrange(embeddings, '(b h w) -> b h w', b=b, h=h, w=w)

        all_embeddings[ptr: ptr + b] = embeddings
        ptr += b

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    print('Saving results')
    np.save(filename, all_embeddings.cpu().numpy())
