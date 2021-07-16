import torchvision
import math

def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)


def update_metrics_dict(dict1, dict2):
    for k, v in dict1.items():
        dict2[k] += v
    return dict2


def write_image_tb(writer, data, iteration, name):
    image = torchvision.utils.make_grid(data, nrow=int(math.sqrt(len(data))))
    image = (image - image.min()) / (image.max() - image.min())
    writer.add_image(name, image, iteration, dataformats='CHW')


def write_images_tb(writer, data, iteration, names=(16, 31)):
    if type(data) == list:
        for d, name in zip(data, names):
            write_image_tb(writer, d, iteration, name)
    else:
        write_image_tb(writer, data, iteration, names[0])