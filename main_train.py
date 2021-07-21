from utils.common_utils import read_config, prepare_run
from modules.moco.builder import ContrastiveModel
from data.data_retriever import ContrastiveDataset
from utils.common_utils import get_train_transformations, get_optimizer, adjust_learning_rate, sample_results, str2bool
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.model_utils import load_checkpoint, load_pretrained_backbone, load_pretrained_aspp
from utils.train_utils import train_epoch


def main():
    writer, device, current_time = prepare_run(root_path, FLAGS.config)
    config['device'] = device
    common_t = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    augment_t = get_train_transformations()

    dataset = ContrastiveDataset(data_path.joinpath('ambulatorium_all.hdf5'), common_transform=common_t,
                                 augment_transform=augment_t, n_classes=config['num_classes'])
    dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=num_workers,
                            drop_last=True)

    model = ContrastiveModel(config)
    model.to(device)
    model.train()

    opt = get_optimizer(config, model.parameters())
    print(f'Chosen optimizer {opt}')

    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

    label_criterion = nn.BCEWithLogitsLoss()

    model = load_pretrained_backbone(config, model, device=device)
    model = load_pretrained_aspp(config, model, device=device)
    model, opt, scaler, start_epoch = load_checkpoint(config, model, opt, scaler, device=device)
    ckpt_path = root_path.joinpath('ckpts', f'{current_time}.pth')

    config['metrics'] = {}
    print(f'Defined metrics {config["metrics"]}')

    config['writing_freq'] = max(1, len(dataset) // (config['train_kwargs']['writing_per_epoch'] * config['train_batch_size']))

    for epoch in range(start_epoch, config['epochs']):
        lr = adjust_learning_rate(config, opt, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        print('Train...')
        model, _ = train_epoch(config, model, dataloader, label_criterion, opt, scaler, writer, epoch)

        print('Sample results...')
        sample_results(model, dataset, config['num_classes'], config['train_kwargs']['saved_images_per_epoch'], device,
                       writer=writer, epoch_num=epoch, debug=True)

        ckpt = {'optimizer': opt.state_dict(), 'model': model.state_dict(), 'epoch': epoch + 1}
        if config['use_amp']:
            ckpt['scaler'] = scaler.state_dict()
        torch.save(ckpt, ckpt_path)



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

    parser.add_argument('-mp', '--mixed_precision',
                        default=True,
                        type=str2bool,
                        help='Use mixed precision')

    FLAGS, unparsed = parser.parse_known_args()
    config = read_config(FLAGS.config)

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[2].joinpath('Datasets')
        num_workers = 0
    else:
        data_path = Path('/storage/homefs/jg20n729/OCT_Detection/Datasets')
        num_workers = 8

    root_path = Path(__file__).resolve().parents[0]
    config['use_amp'] = FLAGS.mixed_precision

    main()