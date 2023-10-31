from pathlib import Path
from typing import Union

import numpy as np
import torch
from torchvision.transforms import transforms, v2
import PIL.Image as Image
from torchvision.transforms.v2 import functional as F


class ImageLoader:
    def __init__(self, image_size, mean_pixel_values=None, std_pixel_values=None):
        if std_pixel_values is None:
            std_pixel_values = [1, 1, 1]
        if mean_pixel_values is None:
            mean_pixel_values = [0, 0, 0]
        self.image_size = image_size
        self.mean_pixel_values = mean_pixel_values
        self.std_pixel_values = std_pixel_values

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __call__(self, image: Union[Path, np.ndarray]):
        transforms = [
            v2.Resize(size=(self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=self.mean_pixel_values, std=self.std_pixel_values),
        ]
        if isinstance(image, Path):
            transforms = [v2.ToImage()] + transforms
            image = Image.open(image).convert('RGB')
        transform = v2.Compose(transforms)
        return transform(image)


##############################
# Dataset dependent values
##############################
IMAGE_SIZE = 80
MEAN_PIXEL_VALUES = [0.4314, 0.4189, 0.3903]
STD_PIXEL_VALUES = [0.1910, 0.1809, 0.1889]
IMAGE_LOADER = ImageLoader(image_size=IMAGE_SIZE,
                           mean_pixel_values=MEAN_PIXEL_VALUES,
                           std_pixel_values=STD_PIXEL_VALUES)
