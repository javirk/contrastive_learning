import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from utils.logs_utils import write_to_tb, update_metrics_dict
from utils.common_utils import IoU_per_class, apply_criterion


def train_step(config, data, model, criterion_dict, optimizer):
    optimizer.zero_grad()

    input_batch = data['images'].to(config['device'])
    transformed_batch = data['transformed_images'].to(config['device'])
    healthy_batch = data['healthy_images'].to(config['device'])
    labels = data['labels'].to(config['device'])

    neg, pos, sal_prediction, sal_target = model(input_batch, transformed_batch, healthy_batch)

    cl_loss = criterion_dict['CL'](pos, neg)
    class_loss = criterion_dict['label'](sal_prediction, sal_target)
    loss = config['train_kwargs']['lambda_cl'] * cl_loss + class_loss

    loss.backward()
    optimizer.step()

    if config['dataset_type'] == 'binary':
        ## Now compute the metrics and return
        y_pred = torch.softmax(sal_prediction, dim=1).argmax(dim=1).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

        m = {}
        for name, metric in config['metrics'].items(): # THIS IS WRONG NOW, but we don't enter here
            if name == 'f1_score':
                m[f'{name}'] = metric(y_true, y_pred, average='macro')  # I think
            else:
                m[f'{name}'] = metric(y_true.astype('uint8'), y_pred)
    else:
        ## Now compute the metrics and return
        y_pred = torch.sigmoid(sal_prediction).detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

        m = {}
        for name, metric in config['metrics'].items():
            if name == 'f1_score':
                # Use a classification threshold of 0.1
                m[f'{name}'] = metric(y_true, y_pred)
            else:
                m[f'{name}'] = metric(y_true.astype('uint8'), y_pred)

    return model, m, loss.item(), cl_loss.item(), pos.mean().item(), neg.mean()


def validation_step(config, data, model, kmeans, criterion, device):
    input_batch = data['images'].to(device)
    gt = data['segmentations']  # The predictions will be later on CPU because of the kmeans.

    with torch.inference_mode():
        pred, _ = model.module.forward_validation(input_batch, kmeans, keep_coarse_bg=True)

    iou_class = IoU_per_class(pred, gt, config['val_kwargs']['k_means']['n_clusters'], 0.5)
    _, mean_iou_fluid, mean_iou_bg = apply_criterion(iou_class, criterion)  # We only need the IoU for this part

    m = {'IoU_fluid': mean_iou_fluid, 'IoU_bg': mean_iou_bg}
    return m


def train_epoch(config, model, loader, criterion_dict, optimizer, writer, epoch_num):
    writing_freq = config['writing_freq']
    model.train()
    running_loss = 0.
    running_clloss = 0.
    running_pos = 0.
    running_neg = 0.
    running_metrics = {k: 0 for k in config['metrics'].keys()}
    for i, data in enumerate(loader):
        outputs = train_step(config, data, model, criterion_dict, optimizer)
        model, metrics_results, loss, cl_loss, positive_sim, negative_sim = outputs

        running_loss += loss
        running_clloss += cl_loss
        running_pos += positive_sim * model.module.T  # So it makes sense to compare similarities
        running_neg += negative_sim * model.module.T
        running_metrics = update_metrics_dict(running_metrics, metrics_results)

        if i % writing_freq == (writing_freq - 1):
            n_iteration = epoch_num * len(loader) + i + 1
            epoch_loss = running_loss / writing_freq
            epoch_clloss = running_clloss / writing_freq
            epoch_pos = running_pos / writing_freq
            epoch_neg = running_neg / writing_freq

            running_metrics = {k: v / writing_freq for k, v in running_metrics.items()}
            running_metrics['loss'] = epoch_loss
            running_metrics['CL_loss'] = epoch_clloss
            running_metrics['Positive'] = epoch_pos
            running_metrics['Negative'] = epoch_neg

            print(f'i={i}, {running_metrics}')
            write_to_tb(writer, running_metrics.keys(), running_metrics.values(), n_iteration, phase=f'train')

            running_loss = 0.
            running_clloss = 0.
            running_pos = 0.
            running_neg = 0.
            running_metrics = {k: 0 for k in config['metrics'].keys()}

    return model, n_iteration


def validate_epoch(config, model, loader, criterion, writer, epoch_num, device):
    model.eval()
    running_metrics = {'IoU_fluid': 0., 'IoU_bg': 0.}
    for data in loader:
        kmeans = KMeans(n_clusters=config['val_kwargs']['k_means']['n_clusters'])
        metrics_results = validation_step(config, data, model, kmeans, criterion, device)

        running_metrics = update_metrics_dict(running_metrics, metrics_results)

    running_metrics = {k: v / len(loader) for k, v in running_metrics.items()}
    write_to_tb(writer, running_metrics.keys(), running_metrics.values(), epoch_num, phase=f'val')
