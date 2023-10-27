# TODO: make PMF call this script if no prototype folder is specified

import itertools
from pathlib import Path

import torch
from torchvision.transforms import transforms
from pmf_helpers import IMAGE_SIZE, PIL_IMAGE_TRANSFORM, pil_loader_rgb
import PIL.Image as Image
import torch.nn as nn
import torch.nn.functional as F
import models.vision_transformer as vit
from scripts.pmf.models import ProtoNet


class ProtoTypeCreator():
    def __init__(self, path_to_dataset):
        self.protonet = ProtoNet(
            backbone=self._get_model_backbone())  # Actually I should only have this file from the repo
        self.protonet.to("cuda:0")
        # TODO: Load from dict

        self.path_to_dataset = path_to_dataset
        # TODO: why init??

    def calculate_prototypes(self, output_path, batch_size=150):
        output_path.mkdir(parents=True, exist_ok=False)
        for class_path in self.path_to_dataset.iterdir():
            print("Processing:", class_path)
            if class_path.is_dir():
                images = [image for image in class_path.iterdir() if image.is_file()]
                images_feature_tensor = torch.empty(size=(len(images), 384),
                                                    dtype=torch.float, device="cuda:0", requires_grad=False)
                single_image_tensor = torch.empty(size=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float, device="cuda:0",
                                                  requires_grad=False)

                images_in = 0
                while images:
                    image_batch, images = images[:batch_size], images[batch_size:]
                    batch_image_tensor = torch.empty(size=(len(image_batch), 3, IMAGE_SIZE, IMAGE_SIZE),
                                                     dtype=torch.float, device="cuda:0", requires_grad=False)
                    for i, image in enumerate(image_batch):
                        image = pil_loader_rgb(image)
                        batch_image_tensor[i] = single_image_tensor.copy_(PIL_IMAGE_TRANSFORM(image))

                    _, C, H, W = batch_image_tensor.shape
                    with torch.no_grad():
                        image_features = self.protonet.backbone.forward(batch_image_tensor.view(-1, C, H, W))
                    images_feature_tensor[images_in:images_in + len(image_batch)] = image_features
                    images_in += len(image_batch)
                prototypes = torch.mean(images_feature_tensor, dim=0)
                torch.save(prototypes, output_path.joinpath(class_path.name + ".pt"))

    @staticmethod
    def _get_model_backbone():
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))
        return model


if __name__ == "__main__":
    path_to_ds = Path(__file__).parent.parent.parent.joinpath("data", "Mini-ImageNet", "test")
    output_path = Path(__file__).parent.parent.parent.joinpath("data", "Mini_ImageNet_Prototypes")
    prototype_creator = ProtoTypeCreator(path_to_dataset=path_to_ds)
    prototype_creator.calculate_prototypes(output_path=output_path)
