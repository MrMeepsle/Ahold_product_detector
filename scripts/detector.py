#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import rospy
import ultralytics
from ahold_product_detection.msg import Detection, RotatedBoundingBox
from ahold_product_detection.srv import *
from cv_bridge import CvBridge
# message and service imports
from sensor_msgs.msg import Image, PointCloud2
from ultralytics.utils.plotting import Annotator

from opencv_helpers import RotatedRect
from rotation_compensation import RotationCompensation


class CameraData:
    def __init__(self, listen_to_pointcloud: bool = False) -> None:
        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)

        self.depth_msg = None
        self.rgb_msg = None
        self.pointcloud_msg = None

        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image, timeout=10)
        rospy.wait_for_message("/camera/color/image_raw", Image, timeout=10)

        if listen_to_pointcloud:
            self.pointcloud_subscriber = rospy.Subscriber("/camera/depth/color/points", PointCloud2,
                                                          self.pointcloud_callback)
            rospy.wait_for_message("/camera/depth/color/points", PointCloud2, timeout=15)

    def depth_callback(self, data):
        self.depth_msg = data

    def pointcloud_callback(self, data):
        self.pointcloud_msg = data

    def rgb_callback(self, data):
        self.rgb_msg = data

    @property
    def data(self):
        # TODO: timesync or check if the time_stamps are not too far apart (acceptable error)
        return self.rgb_msg, self.depth_msg, self.pointcloud_msg, self.rgb_msg.header.stamp


class ProductDetector:
    def __init__(self, rotate: bool, visualize_results: bool, yolo_weights_path: Path) -> None:
        self.rotate = rotate
        self.visualize_results = visualize_results

        self.camera = CameraData()
        self.rotation_compensation = RotationCompensation()
        self.rate = rospy.Rate(30)
        self.bridge = CvBridge()

        self.model = ultralytics.YOLO(yolo_weights_path)

        self.detection_pub = rospy.Publisher("/detection_results", Detection, queue_size=10)

    def plot_detection_results(self, frame, results):
        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

            frame = annotator.result()

            cv2.imshow("Result", frame)
            cv2.waitKey(1)

    @staticmethod
    def show_rotated_results(image, boxes, angle):
        for box in boxes:
            centers_dims = [(int(box[2 * j]), int(box[2 * j + 1])) for j in range(2)]
            RotatedRect(image, centers_dims[0], centers_dims[1][0], centers_dims[1][1], -angle, (0, 0, 255), 2, )
        cv2.imshow("Rotated results", image)
        cv2.waitKey(1)

    @staticmethod
    def generate_detection_message(time_stamp, boxes, scores, labels, rgb_msg, depth_msg):
        detection_msg = Detection()
        detection_msg.header.stamp = time_stamp

        bboxes_list = []
        for bbox, label, score in zip(boxes, labels, scores):
            bbox_msg = RotatedBoundingBox()

            bbox_msg.x = int(bbox[0])
            bbox_msg.y = int(bbox[1])
            bbox_msg.w = int(bbox[2])
            bbox_msg.h = int(bbox[3])
            bbox_msg.label = int(label)
            bbox_msg.score = score

            bboxes_list.append(bbox_msg)

        detection_msg.detections = bboxes_list
        detection_msg.rgb_image = rgb_msg
        detection_msg.depth_image = depth_msg

        return detection_msg

    def run(self):
        try:
            rgb_msg, depth_msg, pointcloud_msg, time_stamp = self.camera.data
        except Exception as e:
            rospy.logerr(f"Couldn't read camera data. Error: %s", e)
            return

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        rotated_rgb_image = self.rotation_compensation.rotate_image(rgb_image, time_stamp) if self.rotate else rgb_image
        results = self.model.predict(source=rotated_rgb_image, show=False, save=False, verbose=False, device=0,
                                     agnostic_nms=True)
        boxes = results[0].boxes.xywh.cpu().numpy()
        if self.rotate:  # Inverse rotate output
            boxes, angle = self.rotation_compensation.rotate_bounding_boxes(boxes, rgb_image)

        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        detection_results_msg = self.generate_detection_message(time_stamp=time_stamp, boxes=boxes, scores=scores,
                                                                labels=labels, rgb_msg=rgb_msg, depth_msg=depth_msg)
        self.detection_pub.publish(detection_results_msg)

        # visualization
        if self.visualize_results:
            if self.rotate:
                self.show_rotated_results(rgb_image, boxes, angle)
            else:
                self.plot_detection_results(rotated_rgb_image, results)


if __name__ == "__main__":
    rospy.init_node("product_detector")
    yolo_weight_path = Path(__file__).parent.parent.joinpath("yolo_model", "just_products_best.pt")
    detector = ProductDetector(rotate=False, visualize_results=True, yolo_weights_path=yolo_weight_path)
    while not rospy.is_shutdown():
        detector.run()
        detector.rate.sleep()
