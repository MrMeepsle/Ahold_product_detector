from pathlib import Path

import torch
from torchvision.transforms import transforms
import PIL.Image as Image


def pil_loader_rgb(path_to_img: Path):
    return Image.open(path_to_img).convert('RGB')


##############################
# Dataset dependent values
##############################
IMAGE_SIZE = 80
MEAN_PIXEL_VALUES = [0.3528, 0.4193, 0.4640]
STD_PIXEL_VALUES = [0.2542, 0.2605, 0.2740]
PIL_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_PIXEL_VALUES, std=STD_PIXEL_VALUES),
])
# NP_ARRAY_TRANSFORM = v2.Compose([
#     v2.Resize(size=(80, 80), antialias=True),
#     v2.ToDtype(torch.float32, scale=True),
#     v2.Normalize(mean=MEAN_PIXEL_VALUES, std=STD_PIXEL_VALUES),
# ])