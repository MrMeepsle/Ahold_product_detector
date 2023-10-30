from pathlib import Path

import torch

import torch.nn.functional as F
from torch import nn

import pmf.models.vision_transformer as vit
from pmf.models import ProtoNet
from pmf_helpers import IMAGE_SIZE, PIL_IMAGE_TRANSFORM, pil_loader_rgb


class StaticProtoNet(ProtoNet):
    def __init__(self, path_to_model_weights: Path):
        backbone = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
        super().__init__(backbone)
        super().load_state_dict(torch.load(path_to_model_weights)['model'])

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
    def __init__(self, protonet_folder_path, feature_extractor, path_to_dataset=None):
        self.protonet_folder_path = protonet_folder_path
        self.feature_extractor = feature_extractor
        self.path_to_dataset = path_to_dataset

    def load_prototypes(self, class_to_find):
        prototype_folder_path = self.protonet_folder_path.joinpath("prototypes")
        if not prototype_folder_path.exists():
            self._calculate_prototypes_to_prototype_folder(prototype_folder_path=prototype_folder_path, batch_size=150)
        return self._load_prototypes_from_folder(prototype_folder_path=prototype_folder_path,
                                                 class_to_find=class_to_find)

    @staticmethod
    def _load_prototypes_from_folder(prototype_folder_path: Path, class_to_find: str,
                                     prototype_feature_size: int = 384, amount_of_prototypes: int = 5):
        class_to_find_tensor = torch.load(prototype_folder_path.joinpath(class_to_find + ".pt"))
        other_prototypes = [prototype_file for prototype_file in prototype_folder_path.iterdir() if
                            prototype_file.is_file() and prototype_file.name != class_to_find + ".pt"]
        other_prototypes_tensor = torch.empty(size=(len(other_prototypes), prototype_feature_size),
                                              dtype=torch.float, device="cuda:0", requires_grad=False)
        for i, class_ in enumerate(other_prototypes):
            other_prototypes_tensor[i] = torch.load(class_)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        class_similarity = cos(class_to_find_tensor, other_prototypes_tensor)
        _, top_indices = torch.topk(class_similarity, amount_of_prototypes - 1)

        prototype_tensor = torch.empty(size=(amount_of_prototypes, prototype_feature_size),
                                       dtype=torch.float, device="cuda:0", requires_grad=False)
        prototype_tensor[0] = class_to_find_tensor
        for i, other_prototype in enumerate(other_prototypes_tensor[top_indices]):
            prototype_tensor[i + 1] = other_prototype

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
                        image_features = self.feature_extractor.forward(batch_image_tensor.view(-1, C, H, W))
                    images_feature_tensor[images_in:images_in + len(image_batch)] = image_features
                    images_in += len(image_batch)
                prototypes = torch.mean(images_feature_tensor, dim=0)
                torch.save(prototypes, prototype_folder_path.joinpath(class_path.name + ".pt"))


class PMF:
    def __init__(self, protonet_folder_path: Path, path_to_dataset: Path):
        self.protonet_folder_path = protonet_folder_path
        model_files = list(self.protonet_folder_path.glob('*.pth'))
        if len(model_files) != 1:
            raise Exception("Please provide a directory with just one .pth file")

        self.protonet = StaticProtoNet(path_to_model_weights=model_files[0])
        self.protonet.to("cuda:0")
        self.prototype_loader = ProtoTypeLoader(protonet_folder_path=protonet_folder_path,
                                                feature_extractor=self.protonet.backbone,
                                                path_to_dataset=path_to_dataset)

    def predict(self, images, cutoff_accuracy):
        with torch.no_grad():
            image_features = self.protonet.backbone.forward(images)
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
    protonet_model_folder = Path(__file__).parent.parent.joinpath("models", "RP2K")
    pmf = PMF(protonet_folder_path=protonet_model_folder, path_to_dataset=dataset_path)
    class_to_find = "n02099601"
    pmf.set_class_to_find(class_to_find)
    images = get_images(class_to_find=class_to_find, path_to_dataset=dataset_path, length=5)
    print(pmf.predict(images, cutoff_accuracy=0.6))
