def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)


def update_metrics_dict(dict1, dict2):
    for k, v in dict1.items():
        dict2[k] += v
    return dict2
