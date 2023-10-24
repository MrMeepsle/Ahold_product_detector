import time
from pathlib import Path
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from torchvision.transforms import v2, transforms

import ultralytics
from ultralytics.utils.plotting import Annotator


class PMFDetector:
    def __init__(self, yolo_weights_path: Path):
        self.yolo_model = ultralytics.YOLO(yolo_weights_path)
        self.bounding_box_conf_threshold = 0.3
        self.device = "cuda:0"
        self.pmf_transforms = v2.Compose([
            v2.Resize(size=(80, 80), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.3528, 0.4193, 0.4640], std=[0.2542, 0.2605, 0.2740]),
        ])

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
        for cx, cy, width, height in bounding_boxes.xywh:
            cropped_image = image[int(cy - height / 2):int(cy + height / 2),
                            int(cx - width / 2):int(cx + width / 2)]
            cropped_image = np.transpose(cropped_image, (2, 0, 1))
            cropped_image = torch.from_numpy(cropped_image).to(self.device)
            cropped_image = self.pmf_transforms(cropped_image)
            cropped_images.append(cropped_image)
        return cropped_images

    def predict(self, color_image: np.ndarray, ros_visualize: bool = False):
        prediction = self.yolo_model.predict(source=color_image, show=False, save=False, verbose=False,
                                             device=self.device, agnostic_nms=True, stream=False)[0]
        cropped_images = self._crop_img_with_bounding_boxes(color_image, prediction.boxes)
        # print(cropped_images)
        # if ros_visualize:
        #     self._plot_detection_results(frame=color_image, detection=prediction)


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
yolo_weights = Path(__file__).parent.joinpath("yolo_model", "just_products_best.pt")
detector = PMFDetector(yolo_weights_path=yolo_weights)
detector.predict(color_image=color_image, ros_visualize=True)
