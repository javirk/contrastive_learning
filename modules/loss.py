import torch
import torch.nn as nn
from einops import rearrange

class ContrastiveLearningLoss(nn.Module):

    def __init__(self):
        super(ContrastiveLearningLoss, self).__init__()


    def forward(self, positive_similarity, negative_similarity):
        """
        Computes the CL loss
        :param positive_similarity: torch.Tensor, sim(z, zM+) Shape: pixels x B
        :param negative_similarity: torch.Tensor, sim(zj, zM-j) Shape: pixels x (queue + batch samples)
        :return:
        """
        positive_similarity = rearrange(positive_similarity, 'p b -> b p')

        den = torch.sum(torch.exp(negative_similarity), dim=-1) + torch.exp(positive_similarity)

        l = - positive_similarity + torch.log(den)

        return l