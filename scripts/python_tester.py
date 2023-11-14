from pathlib import Path
from typing import Optional

import PIL.Image
import pyrealsense2 as rs
import numpy as np
import cv2
import torch

import PIL.Image as Image
import torchvision
import ultralytics
from ultralytics.utils.plotting import Annotator

from scripts.pmf_data_helpers import IMAGE_LOADER, SEEN_COLOR, SEEN_CLASSES, UNSEEN_COLOR, UNSEEN_CLASSES, DEFAULT_COLOR
from scripts.pmf_interface import PMF
from torchvision.utils import save_image


class YoloHelper(ultralytics.YOLO):
    def __init__(self, yolo_weights_path, bounding_box_conf_threshold, device):
        super().__init__(yolo_weights_path)
        self._device = device
        if 0 <= bounding_box_conf_threshold < 1:
            self.bounding_box_conf_threshold = bounding_box_conf_threshold
        else:
            raise Exception("No valid confidence threshold supplied")

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        prediction = \
            super().predict(source=source, stream=stream, predictor=predictor, device=self._device, **kwargs)[0]
        bounding_boxes = prediction.boxes[prediction.boxes.conf > self.bounding_box_conf_threshold]
        cropped_images = self._crop_img_with_bounding_boxes(source, bounding_boxes)
        return cropped_images, bounding_boxes

    @staticmethod
    def _crop_img_with_bounding_boxes(image: Image, bounding_boxes: ultralytics.engine.results.Boxes):
        """
        Crop image with predicted bounding boxes
        """
        multi_image_tensor = torch.empty(
            size=(len(bounding_boxes), 3, IMAGE_LOADER.image_size, IMAGE_LOADER.image_size),
            dtype=torch.float, device="cuda:0",
            requires_grad=False)
        i = 0
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image.crop(
                (int(cx - width / 2), int(cy - height / 2), int(cx + width / 2), int(cy + height / 2)))
            multi_image_tensor[i] = IMAGE_LOADER(cropped_image)
            i += 1
        return multi_image_tensor


class ProductDetectorTest:
    def __init__(self, yolo_weights_path: Path, pmf_model_path: Path, yolo_conf_threshold: float,
                 pmf_conf_threshold: float, dataset_path: Optional[Path] = None, device: str = "cuda:0",
                 reload_prototypes=False):
        self.yolo_model = YoloHelper(yolo_weights_path, yolo_conf_threshold, device=device)
        self.classifier = PMF(pmf_model_path, classification_confidence_threshold=pmf_conf_threshold,
                              image_loader=IMAGE_LOADER, path_to_dataset=dataset_path, device=device,
                              reload_prototypes=reload_prototypes)

    @staticmethod
    def _plot_detection_results(frame: PIL.Image.Image, bounding_boxes, scores, classes):
        """
        Plotting function for showing preliminary detection results for debugging
        """
        annotator = Annotator(np.ascontiguousarray(np.asarray(frame)[:, :, ::-1]), font_size=6)
        labels = [f"{class_[:10]} {score.item():.2f}" for class_, score in zip(classes, scores)]
        colors = [SEEN_COLOR if class_ in SEEN_CLASSES else UNSEEN_COLOR if class_ in UNSEEN_CLASSES else DEFAULT_COLOR
                  for class_ in classes]

        for box, label, color in zip(bounding_boxes, labels, colors):
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            annotator.box_label(b, label, color)

        frame = annotator.result()

        cv2.imshow("Result", frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

    def predict(self, color_image: PIL.Image.Image, debug: bool = False):
        cropped_images, bounding_boxes = self.yolo_model.predict(source=color_image, show=False, save=False,
                                                                 verbose=False, agnostic_nms=True, stream=False)
        scores, classes = self.classifier(cropped_images, debug=debug)
        self._plot_detection_results(frame=color_image, bounding_boxes=bounding_boxes, scores=scores, classes=classes)
        return scores, classes


# DEVICE = True
# if DEVICE:
#     config = rs.config()
#
#     # Get device product line for setting a supporting resolution
#     pipeline = rs.pipeline()
#     pipeline_wrapper = rs.pipeline_wrapper(pipeline)
#     pipeline_profile = config.resolve(pipeline_wrapper)
#     device = pipeline_profile.get_device()
#     device_product_line = str(device.get_info(rs.camera_info.product_line))
#
#     found_rgb = False
#     for s in device.sensors:
#         if s.get_info(rs.camera_info.name) == 'RGB Camera':
#             found_rgb = True
#             break
#     if not found_rgb:
#         print("The demo requires Depth camera with Color sensor")
#         exit(0)
#
#     # if device_product_line == 'L500':
#     #     # Could have to do with this
#     #     config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
#     # else:
#     config.enable_stream(rs.stream.color, format=rs.format.bgr8)  # , 1920, 1080, rs.format.bgr8, 30)
#
#     # Start streaming
#     profile = pipeline.start(config)
#     while True:
#         frames = pipeline.wait_for_frames()
#
#         # Get aligned frames
#         color_frame = frames.get_color_frame()
#         color_image = np.asanyarray(color_frame.get_data())
#         cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
#         cv2.imshow('Align Example', color_image)
#         cv2.imwrite("img1.png", color_image)
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break
#     color_image = Image.fromarray(color_image[..., ::-1])
# else:
#     image_path = Path(__file__).parent.parent.joinpath("data", "Test_Shelves", "4.jpg")
#     color_image = Image.open(image_path).convert('RGB')

image_path = Path(__file__).parent.joinpath("img1.png")
color_image = Image.open(image_path).convert('RGB')
yolo_weights = Path(__file__).parent.parent.joinpath("models", "YOLO_just_products.pt")
pmf_weights = Path(__file__).parent.parent.joinpath("models", "PMF.pth")
dataset_path = Path(__file__).parent.parent.joinpath("data", "Custom-Set_FULL")
detector = ProductDetectorTest(yolo_weights_path=yolo_weights, pmf_model_path=pmf_weights, dataset_path=dataset_path,
                               yolo_conf_threshold=0.3, pmf_conf_threshold=0.5, reload_prototypes=False)
detector.classifier.set_class_to_find("35_Zaanse_Hoeve_Volle_Yoghurt - 8718907039963")
detector.predict(color_image=color_image, debug=True)
