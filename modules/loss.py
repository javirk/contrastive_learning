import torch
import torch.nn as nn

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