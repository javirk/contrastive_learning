import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
from pathlib import Path

from utils.common_utils import read_config
from modules.moco.builder import ContrastiveModel
from data.data_retriever import ContrastiveDataset
import utils.common_utils as u
from utils.model_utils import load_checkpoint, load_pretrained_backbone, load_pretrained_aspp
from evaluation_utils.linear_utils import save_linear_embeddings_to_disk


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveModel(config)
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    model = load_pretrained_backbone(config, model, device=device)
    model = load_pretrained_aspp(config, model, device=device)
    model, _, start_epoch = load_checkpoint(config, model, None, device=device, mode='val')

    model_date = config['checkpoint'].split('.')[0].split('-')[1]
    output_folder = model_date[-6:]

    os.makedirs(Path(__file__).resolve().parents[0].joinpath(f'results'), exist_ok=True)
    os.makedirs(Path(__file__).resolve().parents[0].joinpath(f'results/{output_folder}'), exist_ok=True)
    u.copy_file(FLAGS.config, f'results/{output_folder}/config.yml')

    config['embedding_dir'] = f'results/{output_folder}'

    common_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    augment_t = u.get_val_transformations()
    dataset = ContrastiveDataset(data_path.joinpath('oct_test_all.hdf5'), common_transform=common_t,
                                 augment_transform=augment_t, n_classes=config['num_classes'])
    dataloader = DataLoader(dataset, batch_size=config['val_kwargs']['batch_size'], shuffle=False,
                            num_workers=num_workers)

    save_linear_embeddings_to_disk(config, dataloader, model, device=device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='configurations/config.yml',
                        type=str,
                        help='Path to the config file')

    parser.add_argument('-u', '--ubelix',
                        default=1,
                        type=int,
                        help='Running on ubelix (0 is no)')


    FLAGS, unparsed = parser.parse_known_args()
    config = read_config(FLAGS.config)

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[2].joinpath('Datasets')
        num_workers = 0
    else:
        data_path = Path('/storage/homefs/jg20n729/OCT_Detection/Datasets')
        num_workers = 8

    root_path = Path(__file__).resolve().parents[0]
    config['use_amp'] = False

    if 'TL_' in FLAGS.config and config['checkpoint'] == 'None':
        # This only composes the checkpoint filename
        config['checkpoint'] = FLAGS.config.split('/')[-2][3:] + '.pth'

    main()