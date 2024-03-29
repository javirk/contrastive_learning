import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, upsample, num_classes, ndim, classify_embedding=True, use_classification_head=True,
                 upsample_embedding_mode=False):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone  # This is the encoder
        self.upsample = upsample
        self.upsample_embedding_mode = upsample_embedding_mode
        self.use_classification_head = use_classification_head
        self.classify_embedding = classify_embedding

        self.decoder = decoder

        if self.use_classification_head:  # This is the head after the decoder, it will be used for biomarker detection
            self.classification_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                     nn.Flatten(),
                                                     nn.Linear(ndim, num_classes))
        if self.classify_embedding:  # Are the vectors that come from the encoder classified?
            self.backbone.fc_new = nn.Linear(2048, num_classes)
            del self.backbone.fc


    def forward_embeddings(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')

        x = self.backbone.fc_new(x)
        y = rearrange(x, '(b h w) o -> b o h w', b=b, h=h, w=w)
        return y


    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        return_dict = {}

        x = self.backbone(x)
        if self.classify_embedding:
            cl_embedding = self.forward_embeddings(x)
            if self.upsample:
                cl_embedding = F.interpolate(cl_embedding, size=input_shape, mode=self.upsample_embedding_mode)
            return_dict['cls_emb'] = cl_embedding

        embedding = self.decoder(x)  # ASPP + Conv 1x1

        # Upsample to input resolution
        if self.upsample:
            return_dict['seg'] = F.interpolate(embedding, size=input_shape, mode='bilinear', align_corners=False)
        else:
            return_dict['seg'] = embedding

        # Head
        if self.use_classification_head:
            cl = self.classification_head(embedding)
            return_dict['cls'] = cl

        return return_dict
