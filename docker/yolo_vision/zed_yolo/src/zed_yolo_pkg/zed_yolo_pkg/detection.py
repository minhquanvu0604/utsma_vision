#!/usr/bin/env python3
"""
@file yolo_node.py
@brief ROS2 node for detecting colored cones using YOLOv5 and ZED data, publishing cone positions as a ConeArray message and annotated images.
"""

import rclpy
from rclpy.node import Node
import cv2
import torch
import numpy as np
from tf2_ros import TransformListener, Buffer
import tf_transformations
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from eufs_msgs.msg import ConeArray  # Replace with your actual package name
from cv_bridge import CvBridge
from time import time


class YoloNode(Node):
    """
    @class YoloNode
    @brief A ROS2 node for cone detection and publishing cone positions as a ConeArray message and annotated images.
    """
    def __init__(self):
        """
        @brief Constructor for the YoloNode class.
        Initializes the node, sets up publishers and subscribers, and loads the YOLOv5 model.
        """
        super().__init__('yolo_node')

        # Publishers
        self.pub_cone_array = self.create_publisher(ConeArray, 'cone_array', 10)
        self.raw_image_pub = self.create_publisher(Image, '/bb_image', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/zed/left/image_raw',
            self.image_callback,
            10
        )
        self.point_cloud_sub = self.create_subscription(
            Image,
            '/zed/point_cloud',
            self.point_cloud_callback,
            10
        )

        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLOv5 model loading
        self.model = torch.hub.load(
            '/home/ros2_ws/src/computer_vision/zed_yolo_pkg/zed_yolo_pkg/yolov5',
            'custom',
            '/home/ros2_ws/src/computer_vision/zed_yolo_pkg/zed_yolo_pkg/yolov5/best.pt',
            source='local',
            force_reload=True
        )
        self.bridge = CvBridge()

        # Data buffers
        self.image_data = None  # Buffer for the latest image
        self.point_cloud_data = None  # Buffer for the latest point cloud

        # Debugging variables
        self.last_image_time = 0  # Timestamp of the last received image
        self.last_point_cloud_time = 0  # Timestamp of the last received point cloud

        self.get_logger().info("YoloNode initialized and waiting for image and point cloud data...")

    def image_callback(self, msg):
        """
        @brief Callback for the image topic.
        Converts the ROS Image message to OpenCV format and triggers object detection.

        @param msg The ROS Image message containing the left camera image.
        """
        try:
            self.image_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.last_image_time = time()
            self.get_logger().debug("Image received.")
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def point_cloud_callback(self, msg):
        """
        @brief Callback for the point cloud topic.
        Converts the ROS Image message (point cloud) to OpenCV format for 3D processing.

        @param msg The ROS Image message containing the point cloud.
        """
        try:
            self.point_cloud_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.last_point_cloud_time = time()
            self.get_logger().debug("Point cloud received.")
        except Exception as e:
            self.get_logger().error(f"Failed to convert point cloud: {e}")


    def process_image(self):
        """
        @brief Processes the latest image data and performs YOLOv5 object detection.
        Annotates the image with bounding boxes, labels, and red dots at the center of each bounding box,
        and estimates depth from camera calibration if point clouds are unavailable.
        """
        if self.image_data is None:
            self.get_logger().warn("No image data available yet.")
            return

        # Perform YOLO detection
        results = self.model(self.image_data, size=640)

        # Define camera calibration parameters (replace with actual values)
        focal_length_y = 700.0  # Focal length in pixels (vertical)
        principal_point_x = self.image_data.shape[1] / 2  # Image width center
        principal_point_y = self.image_data.shape[0] / 2  # Image height center
        cone_height_real = 0.3  # Real-world height of a cone in meters

        # Define a color map for cone types (RGB format)
        color_map = {
            0: (0, 0, 255),    # Blue cone
            1: (255, 255, 0),  # Yellow cone
            2: (255, 165, 0),  # Orange cone
            3: (255, 140, 0),  # Big orange cone
            'default': (128, 128, 128)  # Unknown cone
        }

        # Copy image for annotation
        annotated_image = self.image_data.copy()

        # Iterate over detected objects
        for *xyxy, conf, cls in results.xyxy[0]:
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox_height = y_max - y_min

            # Debug bounding box coordinates
            self.get_logger().debug(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")

            # Ensure bounding box is within image dimensions
            if x_min < 0 or y_min < 0 or x_max > annotated_image.shape[1] or y_max > annotated_image.shape[0]:
                self.get_logger().warn(f"Bounding box {x_min, y_min, x_max, y_max} out of image bounds. Skipping.")
                continue

            offset = 0.05
            # Estimate depth using the bounding box height and camera intrinsics
            if bbox_height > 0:  # Avoid division by zero
                depth_estimated = (focal_length_y * cone_height_real) / bbox_height + offset
                self.get_logger().info(f"Estimated depth for cone: {depth_estimated:.2f} meters")
            else:
                self.get_logger().warn("Invalid bounding box height for depth estimation.")
                continue

            # Get the color based on class ID
            class_id = int(cls)
            color_rgb = color_map.get(class_id, color_map['default'])

            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (x_min, y_min),
                (x_max, y_max),
                color_rgb,  # Color
                2           # Thickness
            )

            # Calculate the center of the bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Draw a red dot at the center of the bounding box
            cv2.circle(
                annotated_image,
                (center_x, center_y),  # Center coordinates
                5,                    # Radius of the circle
                (255, 0, 0),          # Red color
                -1                    # Thickness (-1 means filled circle)
            )

            # Add label text with depth information
            label = f"{results.names[class_id]}: {depth_estimated:.2f}m" if class_id < len(results.names) else f"Unknown: {depth_estimated:.2f}m"
            cv2.putText(
                annotated_image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font size
                color_rgb,  # Color
                1           # Thickness
            )

        # Debug: Annotated image is ready
        self.get_logger().info("Annotated image ready for publishing.")

        try:
            # Convert to ROS Image message and publish
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")
            self.raw_image_pub.publish(annotated_image_msg)
            self.get_logger().info("Annotated image published successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")


def main(args=None):
    """
    @brief Main function to initialize and run the YoloNode.
    """
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
