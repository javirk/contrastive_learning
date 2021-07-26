import torch
import numpy as np
import os
from einops import rearrange


@torch.no_grad()
def save_linear_embeddings_to_disk(p, val_loader, model, seed=1234, device='cpu'):
    """
    Inspired from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation
    :param p:
    :param val_loader:
    :param model:
    :param seed:
    :param device:
    :return:
    """
    print('Save linearly classified embeddings to disk ...')
    model.eval()
    ptr = 0
    linear_classifier = model.module.model_q.classification_head[2]
    # Remember that:
    # self.classification_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
    #                                                  nn.Flatten(),
    #                                                  nn.Linear(ndim, num_classes))
    if p['val_kwargs']['coarse_pixels_only']:
        filename = os.path.join(p['embedding_dir'], 'embeddings_linear_coarseonly.npy')
    else:
        filename = os.path.join(p['embedding_dir'], 'embeddings_linear.npy')

    all_embeddings = torch.zeros((len(val_loader.sampler), 496, 512)).to(device)
    for i, batch in enumerate(val_loader):
        qdict = model.module.model_q(batch['images'].to(device))
        features = qdict['seg']

        b, c, h, w = features.shape
        features = rearrange(features, 'b dim h w -> (b h w) dim')  # features: pixels x dim

        if p['val_kwargs']['coarse_pixels_only']:
            coarse = qdict['cls_emb']
            coarse = (torch.softmax(coarse, dim=1) > p['model_kwargs']['coarse_threshold'])
            coarse = coarse.int().argmax(dim=1)  # Prediction of each pixel (coarse). B x H x W
            coarse = (coarse != 0).reshape(-1)  # True/False. B.H.W (pixels)
            coarse_idx = torch.nonzero(coarse).squeeze()
        else:
            coarse_idx = torch.tensor(range(features.shape[0]), device=features.device)

        prototypes = torch.index_select(features, index=coarse_idx, dim=0)  # True pixels x dim

        # Mean-variance normalization? For the image we apply avgpool before
        if p['val_kwargs']['mean_var_normalization']:
            mean, var = torch.std_mean(prototypes, dim=0)
            prototypes = (prototypes - mean) / var

        embeddings_linear = linear_classifier(prototypes)
        embeddings_linear = embeddings_linear.softmax(dim=1).argmax(dim=1).int().cpu()

        embeddings = torch.zeros((b * h * w), dtype=torch.int32)
        embeddings[coarse_idx] = embeddings_linear + 1
        embeddings = rearrange(embeddings, '(b h w) -> b h w', b=b, h=h, w=w)

        all_embeddings[ptr: ptr + b] = embeddings
        ptr += b

        if ptr % 300 == 0:
            print('Computing prototype {}'.format(ptr))

    print('Saving results')
    np.save(filename, all_embeddings.cpu().numpy())
