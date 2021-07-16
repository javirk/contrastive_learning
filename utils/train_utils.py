import torch
from utils.logs_utils import write_to_tb, update_metrics_dict


def train_step(data, model, criterion, optimizer, metrics, device):
    optimizer.zero_grad()

    input_batch = data['images'].to(device)
    transformed_batch = data['transformed_images'].to(device)
    healthy_batch = data['healthy_images'].to(device)
    labels = data['labels'].to(device)

    cl_loss, class_prediction = model(input_batch, transformed_batch, healthy_batch)

    class_loss = criterion(class_prediction, labels)
    loss = cl_loss + class_loss

    loss.backward()
    optimizer.step()

    ## Now compute the metrics and return
    y_pred = torch.sigmoid(class_prediction).detach().cpu().numpy().ravel()
    y_true = labels.detach().cpu().numpy().ravel()

    m = {}
    for name, metric in metrics.items():
        if name == 'f1_score':
            # Use a classification threshold of 0.1
            m[f'{name}'] = metric(y_true > 0, y_pred > 0.1)
        else:
            m[f'{name}'] = metric(y_true.astype('uint8'), y_pred)

    return model, m, loss.item(), cl_loss.item()


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


def train_epoch(model, loader, criterion, optimizer, metrics, writing_freq, writer, epoch_num, device):
    model.train()
    running_loss = 0.
    running_clloss = 0.
    running_metrics = {k: 0 for k in metrics.keys()}
    for i, data in enumerate(loader):
        outputs = train_step(data, model, criterion, optimizer, metrics, device)
        model, metrics_results, loss, cl_loss = outputs

        running_loss += loss
        running_clloss += cl_loss
        running_metrics = update_metrics_dict(running_metrics, metrics_results)

        if i % writing_freq == (writing_freq - 1):
            batch_size = len(data['images'])
            n_epoch = epoch_num * len(loader) + i + 1
            epoch_loss = running_loss / (writing_freq * batch_size)
            epoch_clloss = running_clloss / (writing_freq * batch_size)
            running_metrics = {k: v / writing_freq for k, v in running_metrics.items()}
            running_metrics['loss'] = epoch_loss
            running_metrics['CL_loss'] = epoch_clloss
            print(f'i={i}, {running_metrics}')
            write_to_tb(writer, running_metrics.keys(), running_metrics.values(), n_epoch, phase=f'train')

            running_loss = 0.
            running_clloss = 0.
            running_metrics = {k: 0 for k in metrics.keys()}

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
