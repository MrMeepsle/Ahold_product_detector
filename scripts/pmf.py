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


class StaticProtoNet(ProtoNet):
    def __init__(self, path_to_model_weights: Path):
        backbone = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        super().__init__(backbone)
        super().load_state_dict(str(path_to_model_weights))

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
    def __init__(self):
        a = 1


class PPPPMMMFFF:
    def __init__(self):
        a = 1


class PMF:
    def __init__(self, class_to_find: str, closest_classes: list[str], prototype_folder_path: Path,
                 dataset_path: Path = None):
        self.protonet = StaticProtoNet()
        self.protonet.to("cuda:0")
        # TODO: Load from dict

        self.path_to_dataset = dataset_path
        self.all_classes = [class_to_find] + closest_classes

        self.protonet.update_prototypes(self._load_prototypes(prototype_folder_path))
        # TODO: Make class to find argument

    def _load_prototypes(self, prototype_folder_path):
        if not prototype_folder_path.exists():
            self._calculate_prototypes_to_prototype_folder(prototype_folder_path=prototype_folder_path, batch_size=150)
        return self._load_prototypes_from_folder(prototype_folder_path=prototype_folder_path,
                                                 all_classes=self.all_classes)

    @staticmethod
    def _load_prototypes_from_folder(prototype_folder_path, all_classes):
        prototype_tensor = torch.empty(size=(len(all_classes), 384),
                                       dtype=torch.float, device="cuda:0", requires_grad=False)
        for i, class_ in enumerate(all_classes):
            prototype_tensor[i] = torch.load(prototype_folder_path.joinpath(class_ + ".pt"))
        return prototype_tensor

    def _calculate_prototypes_to_prototype_folder(self, prototype_folder_path, batch_size=150):
        prototype_folder_path.mkdir(parents=True, exist_ok=False)
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
                torch.save(prototypes, prototype_folder_path.joinpath(class_path.name + ".pt"))

    def predict(self, images):
        with torch.no_grad():
            image_features = self.protonet.backbone.forward(images)
            predictions = self.protonet.cos_classifier(image_features)
        return predictions


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
    prototype_path = Path(__file__).parent.parent.parent.joinpath("data", "Mini_ImageNet_Prototypes")
    dataset_path = Path(__file__).parent.parent.parent.joinpath("data", "Mini-ImageNet", "test")
    class_to_find = "n01930112"
    pmf = PMF(class_to_find=class_to_find, closest_classes=["n01981276", "n02099601", "n02110063", "n02110341"],
              prototype_folder_path=prototype_path, dataset_path=dataset_path)
    images = get_images(class_to_find=class_to_find, path_to_dataset=dataset_path, length=5)
    print(pmf.predict(images))
