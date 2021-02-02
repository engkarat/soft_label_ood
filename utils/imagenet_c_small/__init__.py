import numpy as np
from PIL import Image
import torchvision
from .corruptions import *

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate, jpeg_compression,
                    speckle_noise, gaussian_blur, spatter, saturate)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}


def corrupt(x, severity=1, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """
    x = Image.fromarray(x) if not isinstance(x, Image.Image) else x
    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](x, severity)
    elif corruption_number >= 0 and corruption_number <= 18:
        x_corrupted = corruption_tuple[corruption_number](x, severity)
    elif corruption_number == 99:
        x_corrupted = x
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")
    return torchvision.transforms.functional.to_pil_image(np.uint8(x_corrupted))
