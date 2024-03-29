import torch
from utils.logs_utils import write_to_tb, update_metrics_dict


def train_step(config, data, model, criterion_dict, optimizer):
    optimizer.zero_grad()

    input_batch = data['images'].to(config['device'])
    transformed_batch = data['transformed_images'].to(config['device'])
    healthy_batch = data['healthy_images'].to(config['device'])
    labels = data['labels'].to(config['device'])

    neg, pos, class_prediction = model(input_batch, transformed_batch, healthy_batch)

    cl_loss = criterion_dict['CL'](pos, neg)
    class_loss = criterion_dict['label'](class_prediction, labels)
    loss = config['train_kwargs']['lambda_cl'] * cl_loss + class_loss

    loss.backward()
    optimizer.step()

    ## Now compute the metrics and return
    y_pred = torch.sigmoid(class_prediction).detach().cpu().numpy().ravel()
    y_true = labels.detach().cpu().numpy().ravel()

    m = {}
    for name, metric in config['metrics'].items():
        if name == 'f1_score':
            # Use a classification threshold of 0.1
            m[f'{name}'] = metric(y_true > 0, y_pred > 0.1)
        else:
            m[f'{name}'] = metric(y_true.astype('uint8'), y_pred)

    return model, m, loss.item(), cl_loss.item(), pos.mean().item(), neg.mean()


def validation_step(data, model, criterion, metrics, device):
    # TODO: Change all this function. I don't know what validation I should have here because
    # the predicted labels can be very different to the GT ones.
    raise NotImplementedError

    input_batch = data['images'].to(device)
    gt = data['labels'].to(device)

    model.forward_validation(input_batch)

    loss = criterion(out, gt).detach()

    ## Now compute the metrics and return
    y_pred = torch.sigmoid(out).detach().cpu().numpy().ravel()
    y_true = gt.detach().cpu().numpy().ravel()

    m = {}
    for name, metric in metrics.items():
        if name == 'f1_score':
            # Use a classification threshold of 0.1
            m[f'{name}'] = metric(y_true > 0, y_pred > 0.1)
        else:
            m[f'{name}'] = metric(y_true.astype('uint8'), y_pred)

    return m, loss.item()


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
        running_pos += positive_sim
        running_neg += negative_sim
        running_metrics = update_metrics_dict(running_metrics, metrics_results)

        if i % writing_freq == (writing_freq - 1):
            batch_size = len(data['images'])
            n_epoch = epoch_num * len(loader) + i + 1
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
            write_to_tb(writer, running_metrics.keys(), running_metrics.values(), n_epoch, phase=f'train')

            running_loss = 0.
            running_clloss = 0.
            running_pos = 0.
            running_neg = 0.
            running_metrics = {k: 0 for k in config['metrics'].keys()}

    return model, n_epoch


def validate_epoch(model, loader, criterion, metrics, writer, epoch_num, device):
    model.eval()
    running_loss = 0.
    running_metrics = {k: 0 for k in metrics.keys()}
    for data in loader:
        outputs = validation_step(data, model, criterion, metrics, device)
        metrics_results, loss = outputs

        running_loss += loss
        running_metrics = update_metrics_dict(running_metrics, metrics_results)

    running_metrics = {k: v / len(loader) for k, v in running_metrics.items()}
    running_metrics['loss'] = running_loss / len(loader)
    write_to_tb(writer, running_metrics.keys(), running_metrics.values(), epoch_num, phase=f'val')
