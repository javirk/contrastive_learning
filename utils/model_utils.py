import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from modules.models import ContrastiveSegmentationModel
import modules.resnet as resnet
import os
from pathlib import Path


def load_pretrained_backbone(config, model, device='cpu'):
    if config['backbone_kwargs']['pretraining'] == 'imagenet_classification':
        print('Backbone already loaded with Imagenet weights')
    elif config['backbone_kwargs']['pretraining']:
        filename = config['backbone_kwargs']['pretraining']
        print(f'Loading weights from file {filename}')
        pretrained_path = Path(__file__).resolve().parents[1].joinpath('pretrained_models', filename)
        if not os.path.exists(pretrained_path):
            pretrained_path = Path(__file__).resolve().parents[1].joinpath('weights', filename)

        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f'Backbone loaded from {pretrained_path}')
    else:
        print('No backbone loaded')

    return model


def load_pretrained_aspp(config, model, device='cpu'):
    if config['model_kwargs']['pretraining'] == 'imagenet':
        print('Imagenet weights will be loaded for the head')
        state_dict_head = torch.load('pretrained_models/aspp_imagenet.pth', map_location=device)
        model.module.head.load_state_dict(state_dict_head, strict=False)  # strict=False because of last layer classes

    return model


def load_checkpoint(config, model, optimizer, device='cpu'):
    if os.path.exists('ckpts/' + config['checkpoint']):
        filename = config['checkpoint']
        state_dict = torch.load('ckpts/' + filename, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        epoch = state_dict['epoch']
        print(f'Loaded checkpoint {filename}')
    else:
        epoch = 0
        print('No checkpoint loaded')

    return model, optimizer, epoch


def get_model(p):
    if p['model'] == 'deeplab':
        backbone = resnet.__dict__[p['backbone']](pretrained=True, add_head=False,
                                                  replace_stride_with_dilation=[False, True, True])

        # TODO: That 2048 depends on the backbone!
        decoder = DeepLabHead(2048, p['model_kwargs']['ndim'])
        return ContrastiveSegmentationModel(backbone, decoder, p['model_kwargs']['upsample'], p['num_classes'],
                                            p['model_kwargs']['ndim'],
                                            p['model_kwargs']['classify_embedding'],
                                            p['model_kwargs']['use_classification_head'],
                                            upsample_embedding=p['model_kwargs']['upsample'])
    else:
        raise NotImplementedError(f'{p["model"]} not implemented')


def freeze_backbones(model):
    model.module.model_q.backbone.eval()
    model.module.model_k.backbone.eval()

    for param in model.module.model_q.backbone.parameters():
        param.requires_grad = False
    for param in model.module.model_k.backbone.parameters():
        param.requires_grad = False

    return model
