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
    print('Save kmeans embeddings to disk...')
    model.eval()
    ptr = 0
    dataset_name = p[f'val_kwargs']['dataset'].lower()
    if p['val_kwargs']['coarse_pixels_only']:
        filename = os.path.join(p['embedding_dir'], f'embeddings_kmeans_coarseonly_{dataset_name}.npy')
    else:
        filename = os.path.join(p['embedding_dir'], f'embeddings_kmeans_{dataset_name}.npy')

    all_embeddings = torch.zeros((len(val_loader.sampler), 496, 512)).to(device)
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']
        cls = qdict['cls'].sigmoid().cpu()

        b, c, h, w = features.shape   # BATCH SIZE = 1
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

        if p['val_kwargs']['k_means']['use_pca']:
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
        if p['val_kwargs']['k_means']['remove_background_cluster']:
            embeddings[embeddings == (background_cluster + 1)] = 0
        embeddings = rearrange(embeddings, '(b h w) -> b h w', b=b, h=h, w=w)

        all_embeddings[ptr: ptr + b] = embeddings
        ptr += b

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    print('Saving results')
    np.save(filename, all_embeddings.cpu().numpy())


@torch.no_grad()
def save_average_embeddings(p, val_loader, model, seed=1234, device='cpu'):
    """
    For every image, average the embeddings of the coarsely detected pixels, perform the kmeans on the average and save
    :param p:
    :param val_loader:
    :param model:
    :param seed:
    :param device:
    :return:
    """
    print('Save kmeans embeddings to disk...')
    model.eval()
    ptr = 0
    dataset_name = p[f'val_kwargs']['dataset'].lower()

    all_prototypes = torch.zeros((len(val_loader.sampler), 32)).to(device)
    all_sals = torch.zeros((len(val_loader.sampler), 496, 512))
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']

        bs, c, h, w = features.shape
        features = rearrange(features, 'b dim h w -> b dim (h w)')  # features: pixels x dim

        sal = qdict['cls_emb']
        sal = (torch.softmax(sal, dim=1) > p['model_kwargs']['coarse_threshold'])
        sal = sal.int().argmax(dim=1)  # Prediction of each pixel (coarse). B x H x W
        sal_reshaped = sal.reshape(bs, -1, 1)  # Prediction of each pixel (coarse). B x H.W x 1
        sal_reshaped = sal_reshaped * (sal_reshaped != 0).float()  # B x H.W x 1
        prototypes = torch.bmm(features, sal_reshaped).squeeze()  # B x dim

        all_prototypes[ptr: ptr + bs] = prototypes
        all_sals[ptr: ptr + bs, :, :] = (sal != 0).float()
        ptr += bs

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    # perform kmeans
    all_prototypes = all_prototypes.cpu().numpy()
    all_sals = all_sals.cpu().numpy()
    n_clusters = 3
    print('Kmeans clustering to {} clusters'.format(n_clusters))

    print('Starting kmeans with scikit', 'green')
    pca = PCA(n_components=32, whiten=True)
    all_prototypes = pca.fit_transform(all_prototypes)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_prototypes)

    all_predictions = np.zeros(len(prediction_kmeans), 496, 512)
    for i, pred in enumerate(prediction_kmeans):
        prediction = all_sals[i].copy()
        prediction[prediction == 1] = pred + 1
        all_predictions[i] = prediction

    print('Saving results')
    np.save(os.path.join(p['embedding_dir'], f'mean_embeddings_{dataset_name}.npy'))


def train_kmeans(p, val_loader, model, seed=1234, device='cpu'):
    """
    Train a kmeans. Basically the same function as save_average_embeddings
    :param p:
    :param val_loader:
    :param model:
    :param seed:
    :param device:
    :return:
    """
    import pickle
    print('Train a kmeans...')
    model.eval()
    ptr = 0
    dataset_name = p[f'val_kwargs']['dataset'].lower()

    all_prototypes = torch.zeros((len(val_loader.sampler), 32)).to(device)
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']

        bs, c, h, w = features.shape
        features = rearrange(features, 'b dim h w -> b dim (h w)')  # features: pixels x dim

        sal = qdict['cls_emb']
        sal = (torch.softmax(sal, dim=1) > p['model_kwargs']['coarse_threshold'])
        sal = sal.int().argmax(dim=1)  # Prediction of each pixel (coarse). B x H x W
        sal = sal.reshape(bs, -1, 1)  # Prediction of each pixel (coarse). B x H.W x 1
        sal = sal * (sal != 0).float()  # B x H.W x 1
        prototypes = torch.bmm(features, sal).squeeze()  # B x dim

        all_prototypes[ptr: ptr + bs] = prototypes
        ptr += bs

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    # perform kmeans
    all_prototypes = all_prototypes.cpu().numpy()
    n_clusters = 3
    print('Kmeans clustering to {} clusters'.format(n_clusters))

    print('Starting kmeans with scikit', 'green')
    pca = PCA(n_components=32, whiten=True)
    all_prototypes = pca.fit_transform(all_prototypes)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=seed)
    prediction_kmeans = kmeans.fit_predict(all_prototypes)

    with open(f"trained_models/kmeans_{dataset_name}_{p['checkpoint'][:-4]}.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"trained_models/pca_{dataset_name}_{p['checkpoint'][:-4]}.pkl", "wb") as f:
        pickle.dump(pca, f)


@torch.no_grad()
def predict_trained_kmeans(p, val_loader, model, device='cpu'):
    """
    :param p:
    :param val_loader:
    :param model:
    :param device:
    :return:
    """
    import torch.nn as nn
    import pickle
    print('Save kmeans embeddings to disk...')
    model.eval()
    ptr = 0
    dataset_name = p[f'val_kwargs']['dataset'].lower()
    n_clusters = p['num_classes']
    if p['val_kwargs']['coarse_pixels_only']:
        filename = os.path.join(p['embedding_dir'], f'embeddings_trainedkmeans_coarseonly_{dataset_name}.npy')
    else:
        filename = os.path.join(p['embedding_dir'], f'embeddings_trainedkmeans_{dataset_name}.npy')

    # Load the models
    with open(f"trained_models/pca_{dataset_name}_{p['checkpoint'][:-4]}.pkl", "wb") as f:
        pca = pickle.load(f)

    with open(f"trained_models/kmeans_{dataset_name}_{p['checkpoint'][:-4]}.pkl", "wb") as f:
        kmeans = pickle.load(f)

    all_embeddings = torch.zeros((len(val_loader.sampler), 496, 512)).to(device)
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']

        b, c, h, w = features.shape   # BATCH SIZE = 1
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

        prototypes = prototypes.cpu()

        if p['val_kwargs']['k_means']['use_pca']:
            # In the original code they applied PCA before kmeans
            prototypes_pca = pca.transform(prototypes.numpy())
        else:
            prototypes_pca = prototypes.numpy()

        embeddings_kmeans = kmeans.predict(prototypes_pca)

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
        if p['val_kwargs']['k_means']['remove_background_cluster']:
            embeddings[embeddings == (background_cluster + 1)] = 0
        embeddings = rearrange(embeddings, '(b h w) -> b h w', b=b, h=h, w=w)

        all_embeddings[ptr: ptr + b] = embeddings
        ptr += b

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    print('Saving results')
    np.save(filename, all_embeddings.cpu().numpy())
