import rclpy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from eufs_msgs.msg import BoundingBoxes
from std_srvs.srv import SetBool


class CameraVisionServer(rclpy.Node):

    def __init__(self):
        super().__init__('/camera_vision_server')

        # Parameters
        self.running = True
        publish_period = 0.5  # seconds

        # Subscriptions
        self.sub_img_left = self.create_subscription(
            Image,
            '/zed/zed_node/left_raw/image_raw_color',
            self.left_img_zed_callback,
            10
        )
        self.img_left_buffer: Image = None
        self.sub_img_right = self.create_subscription(
            Image,
            '/zed/zed_node/right_raw/image_raw_color',
            self.right_img_zed_callback,
            10
        )
        self.img_right_buffer: Image = None

        # Timer and Publishers
        self.timer = self.create_timer(publish_period, self.pub_data_callback)
        self.pub_filtered_pointcloud = self.create_publisher(
            PointCloud2, 
            '/utsma_vision/filtered_pc', 
            10
        )
        self.pub_bounding_boxes = self.create_publisher(
            BoundingBoxes, 
            '/utsma_vision/bounding_boxes', 
            10
        )

        self.srv = self.create_service(SetBool, '/utsma_vision/enabled', self.enable_service_callback)
        
    def left_img_zed_callback(self, msg: Image):
        self.img_left_buffer = msg
    def right_img_zed_callback(self, msg: Image):
        self.img_right_buffer = msg

    def pub_data_callback(self):
        if (self.running == False):
            return
        if ((self.img_left_buffer == None) or (self.img_right_buffer == None)):
            return
        
        
        left_img = self.img_left_buffer
        right_img = self.img_right_buffer

        pc_msg: PointCloud2
        bounding_boxes_msg: BoundingBoxes


        # Use camera_vision library to get filtered pointcloud and bounding boxes

        self.pub_filtered_pointcloud.publish(pc_msg)
        self.pub_bounding_boxes.publish(bounding_boxes_msg)

    def enable_service_callback(self, request: SetBool.Request, response: SetBool.Response):
        if (request.data == True):
            self.running = True
            response.success = True
            response.message = "Successfully enabled camera_vision_server"
        elif (request.data == False):
            self.running = False
            response.success = True
            response.message = "Successfully disabled camera_vision_server"
        else:
            response.success = False
            response.message = "request.data is not a bool"

        return response




def main():
    rclpy.init()
    node = CameraVisionServer()
    rclpy.spin(node=node)


if __name__ == '__main__':
    main()