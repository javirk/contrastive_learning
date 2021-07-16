# MoCo implementation based upon https://arxiv.org/abs/1911.05722

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import KMeans

from modules.loss import ContrastiveLearningLoss
from utils.model_utils import get_model


class ContrastiveModel(nn.Module):
    def __init__(self, p):
        """
        p: configuration dict
        """
        super(ContrastiveModel, self).__init__()

        self.K = p['moco_kwargs']['K']
        self.m = p['moco_kwargs']['m']
        self.T = p['moco_kwargs']['T']

        self.cl_loss = ContrastiveLearningLoss(reduction=p['loss_kwargs']['reduction'])

        # create the model 
        self.model_q = get_model(p)
        self.model_k = get_model(p)

        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.dim = p['model_kwargs']['ndim']
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.kmeans = KMeans(n_clusters=p['num_classes'])

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_qt, im_k):
        """

        :param im_q: Images. (B x 3 x H x W)
        :param im_qt: Transformed images (B x 3 x H x W)
        :param im_k: Key images (only healthy) (B x 3 x H x W)
        :return:
        """
        batch_size = im_q.size(0)

        qdict = self.model_q(im_q)
        class_prediction = qdict['cls']
        q = qdict['seg']  # queries: B x dim x H x W
        q = nn.functional.normalize(q, dim=1)
        q = rearrange(q, 'b dim h w -> b (h w) dim')  # queries: batch x pixels x dim

        # compute positive prototypes
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            qtdict = self.model_k(im_qt)
            features = qtdict['seg']  # queries transformed: B x dim x H x W
            # Should I normalize here? I think I shouldn't
            features = rearrange(features, 'b d h w -> b d (h w)')
            qt_pred = qtdict['cls_emb']  # predictions of the transformed queries: B x classes x H x W. This comes from
            # coarse embeddings
            qt_pred = torch.softmax(qt_pred, dim=1).argmax(dim=1)  # Prediction of each pixel. B x H x W
            qt_pred = (qt_pred != 0).reshape(batch_size, -1, 1).float()  # True/False. B x H.W x 1
            features = torch.bmm(features, qt_pred)  # B x dim x 1
            features = nn.functional.normalize(features, dim=1)  # B x dim x 1

            # The following doesn't work because there could be an odd number of pixels
            # mask_indexes = torch.nonzero(qt_pred).view(-1)  # This is a vector with N dimensions
            #
            # features = rearrange(features, 'b d h w -> (b h w) d')
            # features = torch.index_select(features, index=mask_indexes, dim=0)  # N x dim, N is the number of
            #                                                                     # non-healthy points
            # features = features.mean(dim=0)  # Positive samples: pixels x dim.

        # compute key prototypes
        with torch.no_grad():  # no gradient to keys
            kdict = self.model_k(im_k)  # keys: N x dim x H x W
            k = kdict['seg']
            k = nn.functional.normalize(k, dim=1)
            k = k.mean(dim=(2, 3))  # N x dim

        positive_similarity = torch.bmm(q, features)  # shape: batch x pixels x 1
        l_batch = torch.matmul(q, k.t())  # shape: batch x pixels x negatives in batch
        negatives = self.queue.clone().detach()  # shape: dim x negatives
        l_mem = torch.matmul(q, negatives)  # shape: batch x pixels x negatives in memory
        negative_similarity = torch.cat([l_batch, l_mem],
                                        dim=-1)  # shape: batch x pixels x (negatives batch + negatives memory)

        # apply temperature
        positive_similarity /= self.T
        negative_similarity /= self.T

        # Calculate loss
        cl_loss = self.cl_loss(positive_similarity, negative_similarity)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return cl_loss, class_prediction

    @torch.no_grad()
    def forward_validation(self, im):
        segmentation = self.model_q(im)['seg']
        b, c, h, w = segmentation.shape
        segmentation = rearrange(segmentation, 'b c h w -> (b h w) c')

        segmentation = self.kmeans.fit_predict(segmentation.numpy())
        segmentation_class = rearrange(torch.from_numpy(segmentation), '(b h w) -> b h w', b=b, h=h, w=w)
        return segmentation_class


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':

    model = ContrastiveModel()