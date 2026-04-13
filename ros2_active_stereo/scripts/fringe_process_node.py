#!/usr/bin/env python3
import os
import cv2
import cupy as cp

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointField
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Bool
import tf2_ros

from ros2_active_stereo.ros2_active_stereo.scripts.fringe_process import Fringe_Process
from ros2_active_stereo.ros2_active_stereo.scripts.SpatialCorrelation_pytorch import PyTorchStereoCorrel 
import torch


class FringeProcess(Node):
    def __init__(self):
        super().__init__('fringe_process_node')

        self.get_logger().info("Node 'fringe_process_node' criado, esperando transições do ciclo de vida...")

        # Parameters to debug process
        self.declare_parameter('image_path', '/tmp/structured-light')
        self.declare_parameter('debug_save', False)
        self.declare_parameter('debug_show', True)    
        self.declare_parameter('yaml_path', '/home/jetson/ros2_ws/src/ros2_fringe_projection/params/SM4.yaml')
        self.declare_parameter('mod_thresh', 50) #cupy 0.07. torch (0-255)
        self.declare_parameter('rad_tresh', 0.06) #threshold for radian difference
        self.declare_parameter('debug_save_points', False)
        self.declare_parameter('save_filename', 'fringe_points')
        self.declare_parameter('camera_frame_id', '/Active/left_camera_link')
        # KDTree parameters
        self.declare_parameter('neighbours', 15)
        self.declare_parameter('radius', 12)

        # State subscriber
        self.state_sub = self.create_subscription(Bool, '/structured_light/state', self.state_callback, 10)
        # Image publisher for visualization in foxglove
        self.abs_phi_left_debug_pub = self.create_publisher(Image, 'debug/abs_phi_left', 10)
        self.abs_phi_right_debug_pub = self.create_publisher(Image, 'debug/abs_phi_right', 10)
        self.mask_left_debug_pub = self.create_publisher(Image, 'debug/mask_left', 10)
        self.mask_right_debug_pub = self.create_publisher(Image, 'debug/mask_right', 10)

        self.stereo_processor = Fringe_Process()
        self.zscan = PyTorchStereoCorrel(yaml_file=self.get_parameter('yaml_path').get_parameter_value().string_value)
        # Construct variables in case disparity point cloud is not available
        self.zmin = -200
        self.zmax = 200

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)



    def state_callbac(self, msg):
        if msg.data:
            self.get_logger().info("Transição para estado 'PROCESSING' detectada. Iniciando processamento de imagens.")
            self.read_images(self.get_parameter('image_path').get_parameter_value().string_value)
            self.control_process = True

    
    def read_images(self, path):
        """
        Read images from a temporary folder where the camera node saves them.
        Args:
            path (str): The base path where the 'left' and 'right' subfolders are located.
        """
        left_path = sorted(os.path.join(path, "left", "*.png"))
        right_path = sorted(os.path.join(path, "right", "*.png"))
        image_left = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in left_path]
        image_right = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in right_path]
        if len(image_left) != len(image_right):
            return self.get_logger().error(f"Number of images are not equal (left {len(image_left)}) and right ({len(image_right)}). Check {path}.")
        for k in range(len(left_path)):
            self.process_images.set_images(image_left[:,:,k], image_right[:,:,k], index=k)

    def process_images(self, debug_show=False):
        
            abs_phi_left, abs_phi_right, modulation_mask_l, modulation_mask_r = self.stereo_processor.calculate_abs_phi_images(save=self.debug_save)

            if debug_show:
                self.debug_show_images(abs_phi_left, abs_phi_right, modulation_mask_l, modulation_mask_r)
                self.get_logger().info("Imagens exibidas.")

            self.get_logger().info("Processamento concluído e imagens publicadas.")
    
    def debug_show_images(self, abs_phi_left, abs_phi_right, modulation_mask_l, modulation_mask_r):
        # Publicar imagens normalizadas apenas para visualização
            self.publish_debug_image(self.abs_phi_left_debug_pub, abs_phi_left, 'Active/left_camera_link')
            self.publish_debug_image(self.abs_phi_right_debug_pub, abs_phi_right, 'Active/left_camera_link')
            self.publish_debug_image(self.mask_left_debug_pub, modulation_mask_l, 'Active/right_camera_link')
            self.publish_debug_image(self.mask_right_debug_pub, modulation_mask_r, 'Active/right_camera_link')


    def publish_debug_image(self, publisher, image_float64, frame_id="camera", normalize=True):
        if normalize:
            img_min = np.nanmin(image_float64)
            img_max = np.nanmax(image_float64)
            if img_max - img_min > 0:
                image_vis = ((image_float64 - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image_vis = np.zeros_like(image_float64, dtype=np.uint8)
        else:
            image_vis = image_float64.astype(np.uint8)

        msg = self.bridge.cv2_to_imgmsg(image_vis, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FringeProcess()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()