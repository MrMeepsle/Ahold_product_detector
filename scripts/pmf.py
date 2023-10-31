import time
from pathlib import Path
from typing import Optional

import torch

import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.transforms import transforms

import pmf.models.vision_transformer as vit
from pmf.models import ProtoNet
from pmf_helpers import IMAGE_SIZE, PIL_IMAGE_TRANSFORM, pil_loader_rgb


class StaticProtoNet(ProtoNet):
    def __init__(self, model: dict):
        backbone = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        super().__init__(backbone)
        super().load_state_dict(model)

        self.prototypes = None

    def update_prototypes(self, prototypes):
        self.prototypes = F.normalize(prototypes, p=2, dim=prototypes.dim() - 1, eps=1e-12)

    def cos_classifier(self, features, **kwargs):
        if self.prototypes is None:
            raise Exception("No prototype set")
        else:
            f = F.normalize(features, p=2, dim=features.dim() - 1, eps=1e-12)
            cls_scores = f @ self.prototypes.transpose(0, 1)
            cls_scores = self.scale_cls * (cls_scores + self.bias)
            cls_scores = self.sigmoid(cls_scores)
            return cls_scores


class ProtoTypeLoader():
    def __init__(self, feature_extractor: torch.nn, image_transform: transforms.Compose,
                 prototype_dict: Optional[dict] = None,
                 path_to_dataset: Optional[Path] = None, ):
        self.feature_extractor = feature_extractor
        self.image_transform = image_transform
        self.prototype_dict = prototype_dict
        if self.prototype_dict is None:
            self.prototype_dict = self.fill_prototype_dict(batch_size=150, path_to_dataset=path_to_dataset)

    def load_prototypes(self, class_to_find: str, amount_of_prototypes: int = 5):
        prototype_dict = self.prototype_dict
        class_to_find_tensor = prototype_dict.pop(class_to_find)
        other_prototypes_tensor = torch.empty(size=(len(prototype_dict), class_to_find_tensor.shape[0]),
                                              dtype=torch.float, device="cuda:0", requires_grad=False)
        for i, key in enumerate(prototype_dict.keys()):
            other_prototypes_tensor[i] = prototype_dict[key]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        class_similarity = cos(class_to_find_tensor, other_prototypes_tensor)
        _, top_indices = torch.topk(class_similarity, amount_of_prototypes - 1)

        prototype_tensor = torch.empty(size=(amount_of_prototypes, class_to_find_tensor.shape[0]),
                                       dtype=torch.float, device="cuda:0", requires_grad=False)
        prototype_tensor[0] = class_to_find_tensor
        for i, other_prototype in enumerate(other_prototypes_tensor[top_indices]):
            prototype_tensor[i + 1] = other_prototype
        return prototype_tensor

    def fill_prototype_dict(self, path_to_dataset, batch_size=150):
        prototype_dict = {}
        if not path_to_dataset.exists():
            raise Exception("Please specify a valid path to dataset")
        for class_path in path_to_dataset.iterdir():
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
                        batch_image_tensor[i] = single_image_tensor.copy_(self.image_transform(image))

                    _, C, H, W = batch_image_tensor.shape
                    with torch.no_grad():
                        image_features = self.feature_extractor.forward(batch_image_tensor.view(-1, C, H, W))
                    images_feature_tensor[images_in:images_in + len(image_batch)] = image_features
                    images_in += len(image_batch)
                class_prototype = torch.mean(images_feature_tensor, dim=0)
                prototype_dict[class_path.name] = class_prototype
        return prototype_dict


class PMF:
    def __init__(self, pmf_model_path: Path, image_transform: torchvision.transforms,
                 path_to_dataset: Optional[Path] = None):
        self.pmf_path = pmf_model_path
        pmf_dict = torch.load(self.pmf_path)
        reload_prototypes = False

        if 'transforms' not in pmf_dict:
            reload_prototypes = True
        else:
            for i, j in zip(pmf_dict['transforms'].transforms, image_transform.transforms):
                if i.__dict__ != j.__dict__:
                    reload_prototypes = True
        self.image_transform = image_transform

        if "model" not in pmf_dict:
            raise Exception("Please provide path to a model")
        self.protonet = StaticProtoNet(model=pmf_dict["model"])
        self.protonet.to("cuda:0")

        if "prototype_dict" not in pmf_dict or reload_prototypes is True:
            self.prototype_loader = ProtoTypeLoader(feature_extractor=self.protonet.backbone,
                                                    path_to_dataset=path_to_dataset,
                                                    image_transform=self.image_transform)
            self.save_model_dict(self.pmf_path)
        else:
            self.prototype_loader = ProtoTypeLoader(feature_extractor=self.protonet.backbone,
                                                    prototype_dict=pmf_dict["prototype_dict"],
                                                    image_transform=image_transform)

    def save_model_dict(self, path: Path):
        print("Saving prototype loader")
        dict_to_save = {"model": self.protonet.state_dict(),
                        "prototype_dict": self.prototype_loader.prototype_dict,
                        "transforms": self.image_transform}
        torch.save(dict_to_save, path)

    def predict(self, image_tensors, cutoff_accuracy):
        with torch.no_grad():
            image_features = self.protonet.backbone.forward(image_tensors)
            predictions = self.protonet.cos_classifier(image_features)
        values, indices = torch.max(predictions, dim=1)
        return torch.logical_and((values >= cutoff_accuracy), (indices == 0))

    def set_class_to_find(self, class_to_find):
        self.protonet.update_prototypes(self.prototype_loader.load_prototypes(class_to_find))


def get_images(class_to_find: str, length: int, path_to_dataset: Path):
    multi_image_tensor = torch.empty(size=(length, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float, device="cuda:0",
                                     requires_grad=False)
    single_image_tensor = torch.empty(size=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float, device="cuda:0",
                                      requires_grad=False)
    path_to_class = path_to_dataset.joinpath(class_to_find)
    images = [image for image in path_to_class.iterdir() if image.is_file()][:length]
    for i, image in enumerate(images):
        pil_image = pil_loader_rgb(image)
        multi_image_tensor[i] = single_image_tensor.copy_(PIL_IMAGE_TRANSFORM(pil_image))
    return multi_image_tensor


if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.joinpath("data", "Mini-ImageNet", "test")
    protonet_model = Path(__file__).parent.parent.joinpath("models", "RP2K", "RP2KProtoNet.pth")
    pmf = PMF(pmf_model_path=protonet_model, path_to_dataset=dataset_path, image_transform=PIL_IMAGE_TRANSFORM)
    class_to_find = "n02099601"
    pmf.set_class_to_find(class_to_find)
    images = get_images(class_to_find=class_to_find, path_to_dataset=dataset_path, length=5)
    pmf.predict(images, cutoff_accuracy=0.6)
