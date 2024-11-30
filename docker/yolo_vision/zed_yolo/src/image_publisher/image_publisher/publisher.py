import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self, image_path):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/zed/left/image_raw', 10)
        self.bridge = CvBridge()
        self.image_path = image_path

        # Validate the image path and load the image once
        if not os.path.exists(self.image_path):
            self.get_logger().error(f"Image not found at {self.image_path}")
            return

        self.cv_image = cv2.imread(self.image_path)
        if self.cv_image is None:
            self.get_logger().error(f"Failed to load image: {self.image_path}")
            return

        # Convert the image to RGB format
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        # Create a timer to publish every 1 second
        self.timer = self.create_timer(1.0, self.publish_image)

    def publish_image(self):
        if self.cv_image is None:
            self.get_logger().error("No valid image to publish")
            return

        # Convert to ROS2 Image message
        ros_image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding="rgb8")
        self.publisher_.publish(ros_image)
        self.get_logger().info(f"Published image from {self.image_path}")


def main(args=None):
    rclpy.init(args=args)

    # Path to your PNG image
    # image_path = '/home/ros2_ws/src/computer_vision/image_publisher/resource/image_2.jpg'
    image_path = '/home/ros2_ws/src/computer_vision/image_publisher/resource/image_1.png'

    image_publisher = ImagePublisher(image_path)

    rclpy.spin(image_publisher)

    image_publisher.destroy_node()
    rclpy.shutdown()
