import torch
import torch.nn as nn

class ContrastiveLearningLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(ContrastiveLearningLoss, self).__init__()
        self.reduction = reduction

    def forward(self, positive_similarity, negative_similarity):
        """
        Computes the CL loss
        :param positive_similarity: torch.Tensor, sim(z, zM+) Shape: B x pixels x 1
        :param negative_similarity: torch.Tensor, sim(zj, zM-j) Shape: B x pixels x (queue + batch samples)
        :return:
        """
        positive_similarity = positive_similarity.squeeze(-1)

        den = torch.sum(torch.exp(negative_similarity), dim=-1)

        l = - positive_similarity + torch.log(den)

        if self.reduction == 'mean':
            return torch.mean(l)
        else:
            return l