from pathlib import Path

import pyrealsense2 as rs
import numpy as np
import cv2
import torch

import ultralytics
from ultralytics.utils.plotting import Annotator

from scripts.pmf_helpers import IMAGE_LOADER
from scripts.pmf_interface import PMF


class PMFDetector:
    def __init__(self, yolo_weights_path: Path, pmf_model_path: Path):
        self.yolo_model = ultralytics.YOLO(yolo_weights_path)
        self.classifier = PMF(pmf_model_path=pmf_model_path, image_loader=IMAGE_LOADER)
        self.bounding_box_conf_threshold = 0.3
        self.device = "cuda:0"

    def _plot_detection_results(self, frame, detection):
        """
        Plotting function for showing preliminary detection results for debugging
        """
        annotator = Annotator(frame)
        boxes = detection.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, self.yolo_model.names[int(c)])

        frame = annotator.result()

        cv2.imshow("Result", frame)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()

    def _crop_img_with_bounding_boxes(self, image: np.ndarray, bounding_boxes: ultralytics.engine.results.Boxes):
        """
        Crop image with predicted bounding boxes
        """
        cropped_images = []
        bounding_boxes = bounding_boxes[bounding_boxes.conf > self.bounding_box_conf_threshold]
        multi_image_tensor = torch.empty(
            size=(len(bounding_boxes), 3, IMAGE_LOADER.image_size, IMAGE_LOADER.image_size),
            dtype=torch.float, device="cuda:0",
            requires_grad=False)
        i = 0
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image[int(cy - height / 2):int(cy + height / 2),
                            int(cx - width / 2):int(cx + width / 2)]
            cropped_image = np.transpose(cropped_image, (2, 0, 1))
            cropped_image = torch.from_numpy(cropped_image).to(self.device)
            multi_image_tensor[i] = IMAGE_LOADER(cropped_image)
            i += 1
        return multi_image_tensor

    def predict(self, color_image: np.ndarray, ros_visualize: bool = False):
        prediction = self.yolo_model.predict(source=color_image, show=False, save=False, verbose=False,
                                             device=self.device, agnostic_nms=True, stream=False)[0]
        cropped_images = self._crop_img_with_bounding_boxes(color_image, prediction.boxes)
        if ros_visualize:
            self._plot_detection_results(frame=color_image, detection=prediction)
        return self.classifier.predict(cropped_images, cutoff_accuracy=0.5)


DEVICE = False
if DEVICE:
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline = rs.pipeline()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    while True:
        frames = pipeline.wait_for_frames()

        # Get aligned frames
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', color_image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
else:
    image_path = Path(__file__).parent.parent.joinpath("data", "Test_Shelves", "1.jpg")
    color_image = np.asarray(cv2.imread(str(image_path)))

yolo_weights = Path(__file__).parent.parent.joinpath("yolo_model", "just_products_best.pt")
pmf_weights = Path(__file__).parent.parent.joinpath("models", "RP2K", "RP2KProtoNet.pth")
detector = PMFDetector(yolo_weights_path=yolo_weights, pmf_model_path=pmf_weights)
detector.classifier.set_class_to_find("4_AH_Fijngesneden_Tomaten")
print(detector.predict(color_image=color_image, ros_visualize=True))
