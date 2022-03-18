import torch
import torch.nn as nn
import numpy as np

class ContrastiveLearningLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(ContrastiveLearningLoss, self).__init__()
        self.reduction = reduction

    def forward(self, positive_similarity, negative_similarity):
        """
        Computes the CL loss.

        True pixels should be the same number as (positive_similarity == 0).any(0).sum()
        :param positive_similarity: torch.Tensor, sim(z, zM+) Shape: true pixels
        :param negative_similarity: torch.Tensor, sim(zj, zM-j) Shape: true pixels x (queue + batch samples)
        :return:
        """

        den = torch.sum(torch.exp(negative_similarity), dim=-1) + torch.exp(positive_similarity)

        l = - positive_similarity + torch.log(den)

        if self.reduction == 'mean':
            return l.mean()
        else:
            return l


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, labels, void_pixels=None):
        assert (output.size() == labels.size())

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(labels.size()))
        elif self.batch_average:
            final_loss /= labels.size()[0]

        return final_loss
