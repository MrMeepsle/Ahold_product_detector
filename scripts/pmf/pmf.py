import itertools
from pathlib import Path

import torch
from torchvision.transforms import transforms
import PIL.Image as Image
import torch.nn as nn
import torch.nn.functional as F
import models.vision_transformer as vit
from scripts.pmf.models import ProtoNet
from pmf_helpers import IMAGE_SIZE, PIL_IMAGE_TRANSFORM, pil_loader_rgb


class PMF():
    def __init__(self, class_to_find: str, closest_classes: list[str], prototype_folder_path: Path,
                 dataset_path: Path = None):
        self.protonet = ProtoNet(
            backbone=self._get_model_backbone())  # Actually I should only have this file from the repo
        self.protonet.to("cuda:0")
        # TODO: Load from dict

        self.path_to_dataset = dataset_path
        self.prototype_folder_path = prototype_folder_path
        self.all_classes = [class_to_find] + closest_classes

        self.prototypes = self._load_prototypes()
        print(self.prototypes)

    def _load_prototypes(self):
        if not self.prototype_folder_path.exists():
            self._calculate_prototypes_to_prototype_folder(batch_size=150)
        return self._load_prototypes_from_folder(self.prototype_folder_path, self.all_classes)

    @staticmethod
    def _load_prototypes_from_folder(prototype_folder_path, all_classes):
        prototype_tensor = torch.empty(size=(len(all_classes), 384),
                                       dtype=torch.float, device="cuda:0", requires_grad=False)
        for i, class_ in enumerate(all_classes):
            prototype_tensor[i] = torch.load(prototype_folder_path.joinpath(class_ + ".pt"))
        return prototype_tensor

    @staticmethod
    def _get_model_backbone():
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))
        return model

    def forward(self, images):
        with torch.no_grad():
            feat = self.protonet.backbone.forward(x.view(-1, C, H, W))
            feat = feat.view(B, x.shape[1], -1)  # B, nQry, d
            predictions = self.protonet.cos_classifier(self.prototypes, feat)  # B, nQry, nC
        return predictions

    def _calculate_prototypes_to_prototype_folder(self, batch_size=150):
        self.prototype_folder_path.mkdir(parents=True, exist_ok=False)
        if not self.path_to_dataset.exists():
            raise Exception("Cannot create Prototypes without a provided dataset path!!")
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
                torch.save(prototypes, self.prototype_folder_path.joinpath(class_path.name + ".pt"))


if __name__ == "__main__":
    prototype_path = Path(__file__).parent.parent.parent.joinpath("data", "Mini_ImageNet_Prototypes")
    dataset_path = Path(__file__).parent.parent.parent.joinpath("data", "Mini-ImageNet", "test")
    PMF(class_to_find="n01930112", closest_classes=["n01981276", "n02099601", "n02110063", "n02110341"],
        prototype_folder_path=prototype_path, dataset_path=dataset_path)
