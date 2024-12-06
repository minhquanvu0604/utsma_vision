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
# import tf_transformations
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Point
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance

from cv_bridge import CvBridge
from time import time


class YoloNode(Node):
    """
    @class YoloNode
    @brief A ROS2 node for cone detection and publishing cone positions as a ConeArray message and annotated images.
    """

    # Define a color map for cone types (RGB format)
    COLOR_MAP = {
        0: (0, 0, 255),    # Blue cone
        1: (255, 255, 0),  # Yellow cone
        2: (255, 165, 0),  # Orange cone
        3: (255, 140, 0),  # Big orange cone
        'default': (128, 128, 128)  # Unknown cone
    }

    IMAGE_WIDTH = 1280

    def __init__(self):
        """
        @brief Constructor for the YoloNode class.
        Initializes the node, sets up publishers and subscribers, and loads the YOLOv5 model.
        """
        super().__init__('yolo_node')

        # Publishers
        self.pub_cone_array = self.create_publisher(ConeArrayWithCovariance, '/bb_cone_array', 10)
        self.raw_image_pub = self.create_publisher(Image, '/bb_image', 10)

        # The depth map from the ZED ROS wrapper is usually aligned with the left camera
        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )

        # TODO: use depth image instead
        # shape of the pointcloud is (1228800, 3) which is a flat list of points, each point has 3 coordinates (x, y, z)
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/zed/zed_node/point_cloud/cloud_registered',
            self.pointcloud_callback,
            10
        )
        # self.depth_sub = self.create_subscription(
        #     Image,
        #     '/zed2/depth/depth_registered',  # Replace with your depth topic
        #     self.depth_callback,
        #     10
        # )


        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # YOLOv5 model loading 
        try:
            self.model = torch.hub.load(
                '/root/ros2_ws/src/utsma_vision/model_inference/coord_light_yolov5/scripts/yolov5',
                'custom',
                '/root/ros2_ws/src/utsma_vision/model_inference/coord_light_yolov5/scripts/yolov5/best.pt',
                source='local',
                force_reload=True
            )
            # model_path = '/root/ros2_ws/src/utsma_vision/model_inference/coord_light_yolov5/scripts/best.pt'
            # self.model = torch.load(model_path, map_location=torch.device('cuda'))
            self.model.eval()  

            self.get_logger().info("YOLOv5 model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

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
            self.process_image()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def pointcloud_callback(self, msg):
        self.get_logger().info("PointCloud2 data received.")
        
        # Convert PointCloud2 to numpy array and store in the class variable
        self.point_cloud_data = self.pointcloud2_to_array(msg)

        # self.get_logger().info(f"Point cloud data type: {type(self.point_cloud_data)}")
        # self.get_logger().info(f"Point cloud shape: {self.point_cloud_data.shape}")
        # self.get_logger().info(f"Point cloud sample: {self.point_cloud_data[:5]}")  # Print the first few points

    @staticmethod
    def pointcloud2_to_array(msg):
        """
        Convert a ROS2 PointCloud2 message to a numpy array.
        """
        dtype_list = [
            ('x', np.float32), ('y', np.float32), ('z', np.float32)
        ]
        data = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
        points = np.column_stack((data['x'], data['y'], data['z']))
        return points

    def process_image(self):
        """
        @brief Processes the latest image data and performs YOLOv5 object detection.
        Annotates the image with bounding boxes, labels, and red dots at the center of each bounding box,
        retrieves depth using point cloud data, and publishes cone positions in the camera frame.
        """
        if self.image_data is None:
            self.get_logger().warn("No image data available yet.")
            return

        if self.point_cloud_data is None:
            self.get_logger().warn("No point cloud data available yet.")
            return

        # Perform YOLO detection
        results = self.model(self.image_data, size=640)

        # Initialize ConeArray message
        cone_array_msg = ConeArrayWithCovariance()
        cone_array_msg.header = Header()
        cone_array_msg.header.stamp = self.get_clock().now().to_msg()
        cone_array_msg.header.frame_id = 'zed_left_camera_frame'

        # Copy image for annotation
        annotated_image = self.image_data.copy()

        # Iterate over detected objects
        for *xyxy, conf, cls in results.xyxy[0]:
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox_height = y_max - y_min

            # Ensure bounding box is within image dimensions
            if x_min < 0 or y_min < 0 or x_max > annotated_image.shape[1] or y_max > annotated_image.shape[0]:
                self.get_logger().warn(f"Bounding box {x_min, y_min, x_max, y_max} out of image bounds. Skipping.")
                continue

            # # Calculate the center of the bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            self.get_logger().info(f"Center X: {center_x}, Center Y: {center_y}")

            # Retrieve depth from point cloud with smoothing
            try:
                kernel_size = 5  # Define the size of the smoothing kernel (e.g., 3x3, 5x5, etc.)
                half_k = kernel_size // 2

                # Initialize a list to hold valid depths
                valid_depths = []
                
                # Loop through the kernel region
                for dy in range(-half_k, half_k + 1):
                    for dx in range(-half_k, half_k + 1):
                        px = center_x + dx
                        py = center_y + dy

                        # Check if the coordinates are within bounds
                        if px < 0 or px >= YoloNode.IMAGE_WIDTH or py < 0 or py >= len(self.point_cloud_data):
                            self.get_logger().warn(f"Out of bounds: px={px}, py={py}")
                            continue

                        # Compute the flat index
                        index = py * YoloNode.IMAGE_WIDTH + px

                        if index < 0 or index >= self.point_cloud_data.shape[0]:
                            self.get_logger().warn(f"Invalid index {index}")
                            continue

                        # Access the point
                        point = self.point_cloud_data[index]
                        x, y, z = point  # Unpack x, y, z coordinates

                        # Check validity of depth (z-coordinate)
                        if np.isfinite(z) and z > 0:
                            valid_depths.append(z)  # Append valid depth
                            self.get_logger().info(f"Valid point at index {index}: {x}, {y}, {z}")
                        else:
                            self.get_logger().warn(f"Invalid depth value at index {index}: {z}")

                # Check if we have valid depths
                if valid_depths:
                    Z_cam = np.mean(valid_depths)  # Compute the average depth
                    self.get_logger().info(f"Smoothed depth: {Z_cam:.2f} meters")

                    # Retrieve the corresponding X and Y coordinates
                    index_center = center_y * YoloNode.IMAGE_WIDTH + center_x
                    X_cam = self.point_cloud_data[index_center, 0]
                    Y_cam = self.point_cloud_data[index_center, 1]
                else:
                    raise ValueError("No valid depth values found in the region.")

            except IndexError:
                self.get_logger().warn(f"Point cloud index out of range for ({center_x}, {center_y}). Skipping.")
            except ValueError as e:
                self.get_logger().warn(f"Depth error: {e}")
            except Exception as e:
                self.get_logger().error(f"Unexpected error: {e}")


            # Comment out the world coordinate transformation
            # try:
            #     trans = self.tf_buffer.lookup_transform('world', 'camera_frame', rclpy.time.Time())
            #     translation = [
            #         trans.transform.translation.x,
            #         trans.transform.translation.y,
            #         trans.transform.translation.z
            #     ]
            #     rotation = [
            #         trans.transform.rotation.x,
            #         trans.transform.rotation.y,
            #         trans.transform.rotation.z,
            #         trans.transform.rotation.w
            #     ]
            #     transform_matrix = tf_transformations.concatenate_matrices(
            #         tf_transformations.translation_matrix(translation),
            #         tf_transformations.quaternion_matrix(rotation)
            #     )
            #     camera_coords = np.array([X_cam, Y_cam, Z_cam, 1.0])  # Homogeneous coordinates
            #     world_coords = np.dot(transform_matrix, camera_coords)
            #     X_world, Y_world, Z_world = world_coords[:3]
            # except Exception as e:
            #     self.get_logger().warn(f"Failed to transform to world coordinates: {e}")
            #     continue

            cone = ConeWithCovariance()
            cone.point.x = X_cam
            cone.point.y = Y_cam
            cone.point.z = Z_cam

            class_id = int(cls)
            if class_id == 0:
                cone_array_msg.blue_cones.append(cone)
            elif class_id == 1:
                cone_array_msg.yellow_cones.append(cone)
            elif class_id == 2:
                cone_array_msg.orange_cones.append(cone)
            elif class_id == 3:
                cone_array_msg.big_orange_cones.append(cone)
            else:
                cone_array_msg.unknown_color_cones.append(cone)

            # Annotate the image with bounding boxes and metadata
            color_rgb = YoloNode.COLOR_MAP.get(class_id, YoloNode.COLOR_MAP['default'])
            cv2.rectangle(
                annotated_image,
                (x_min, y_min),
                (x_max, y_max),
                color_rgb,  # Color
                2           # Thickness
            )
            label = f"{results.names[class_id]}: {Z_cam:.2f}m"
            cv2.putText(
                annotated_image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,        # Font size
                color_rgb,  # Color
                1           # Thickness
            )
            self.get_logger().info(f"Camera coordinates: X={X_cam:.2f}, Y={Y_cam:.2f}, Z={Z_cam:.2f}")

            # Annotate the image with a red dot
            cv2.circle(
                annotated_image,
                (center_x, center_y),  # Center coordinates
                5,                    # Radius of the circle
                (255, 0, 0),          # Red color
                -1                    # Thickness (-1 means filled circle)
            )

        # Publish the ConeArray message
        self.pub_cone_array.publish(cone_array_msg)
        self.get_logger().info("ConeArray message published.")

        # Publish the annotated image
        try:
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")
            self.raw_image_pub.publish(annotated_image_msg)
            self.get_logger().info("Annotated image published successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")


    # def process_image(self):
    #     """
    #     @brief Processes the latest image data and performs YOLOv5 object detection.
    #     Annotates the image with bounding boxes, labels, and red dots at the center of each bounding box,
    #     and estimates depth from camera calibration if point clouds are unavailable.
    #     """
    #     if self.image_data is None:
    #         self.get_logger().warn("No image data available yet.")
    #         return

    #     # Perform YOLO detection
    #     results = self.model(self.image_data, size=640)

    #     # Define camera calibration parameters (replace with actual values)
    #     focal_length_y = 700.0  # Focal length in pixels (vertical)
    #     principal_point_x = self.image_data.shape[1] / 2  # Image width center
    #     principal_point_y = self.image_data.shape[0] / 2  # Image height center
    #     cone_height_real = 0.3  # Real-world height of a cone in meters

    #     # Define a color map for cone types (RGB format)
    #     color_map = {
    #         0: (0, 0, 255),    # Blue cone
    #         1: (255, 255, 0),  # Yellow cone
    #         2: (255, 165, 0),  # Orange cone
    #         3: (255, 140, 0),  # Big orange cone
    #         'default': (128, 128, 128)  # Unknown cone
    #     }

    #     # Copy image for annotation
    #     annotated_image = self.image_data.copy()

    #     # Iterate over detected objects
    #     for *xyxy, conf, cls in results.xyxy[0]:
    #         # Bounding box coordinates
    #         x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    #         bbox_height = y_max - y_min

    #         # Debug bounding box coordinates
    #         self.get_logger().debug(f"Bounding box: ({x_min}, {y_min}), ({x_max}, {y_max})")

    #         # Ensure bounding box is within image dimensions
    #         if x_min < 0 or y_min < 0 or x_max > annotated_image.shape[1] or y_max > annotated_image.shape[0]:
    #             self.get_logger().warn(f"Bounding box {x_min, y_min, x_max, y_max} out of image bounds. Skipping.")
    #             continue

    #         offset = 0.05
    #         # Estimate depth using the bounding box height and camera intrinsics
    #         if bbox_height > 0:  # Avoid division by zero
    #             depth_estimated = (focal_length_y * cone_height_real) / bbox_height + offset
    #             self.get_logger().info(f"Estimated depth for cone: {depth_estimated:.2f} meters")
    #         else:
    #             self.get_logger().warn("Invalid bounding box height for depth estimation.")
    #             continue

    #         # Get the color based on class ID
    #         class_id = int(cls)
    #         color_rgb = color_map.get(class_id, color_map['default'])

    #         # Draw bounding box
    #         cv2.rectangle(
    #             annotated_image,
    #             (x_min, y_min),
    #             (x_max, y_max),
    #             color_rgb,  # Color
    #             2           # Thickness
    #         )

    #         # Calculate the center of the bounding box
    #         center_x = (x_min + x_max) // 2
    #         center_y = (y_min + y_max) // 2

    #         # Draw a red dot at the center of the bounding box
    #         cv2.circle(
    #             annotated_image,
    #             (center_x, center_y),  # Center coordinates
    #             5,                    # Radius of the circle
    #             (255, 0, 0),          # Red color
    #             -1                    # Thickness (-1 means filled circle)
    #         )

    #         # Add label text with depth information
    #         label = f"{results.names[class_id]}: {depth_estimated:.2f}m" if class_id < len(results.names) else f"Unknown: {depth_estimated:.2f}m"
    #         cv2.putText(
    #             annotated_image,
    #             label,
    #             (x_min, y_min - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,  # Font size
    #             color_rgb,  # Color
    #             1           # Thickness
    #         )

    #     # Debug: Annotated image is ready
    #     self.get_logger().info("Annotated image ready for publishing.")

    #     try:
    #         # Convert to ROS Image message and publish
    #         annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="rgb8")
    #         self.raw_image_pub.publish(annotated_image_msg)
    #         self.get_logger().info("Annotated image published successfully.")
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to publish annotated image: {e}")


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
