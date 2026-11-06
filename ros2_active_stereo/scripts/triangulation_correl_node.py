#!/usr/bin/env python3
import os
import cv2
import datetime
import numpy as np
import time
import struct
from pipeline_utils import ScanWorkerMixin, validate_pairs

from SpatioCorrelation import PyTorchStereoCorrel as SpatialCorrelator
from zscan_reconstruction import reconstruct_laser

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
from ros2_active_stereo_msgs.srv import ProcessFolder

from std_srvs.srv import Trigger, Empty
from std_msgs.msg import Int16, String
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import tf_transformations

# retirar metodos que foram passados para cpp
# importar as imagens direto da ram (salvas pelo inv_correlation_node.cpp)
# handshake para importar as imagens


class InverseTriangulationNode(ScanWorkerMixin, Node):
    def __init__(self):
        super().__init__('inverse_correlation_node')
        self.get_logger().info('Inverse Correlation Node has been started.')

        # Parameters declaration
        self.declare_parameter('max_block_voxels', 500000)
        self.declare_parameter('reconstruct_laser', True)
        self.declare_parameter('laser_blur', 3)
        self.declare_parameter('surface_min_points', 32)
        self.declare_parameter('depth_band_margin', 6.0)
        self.declare_parameter('split_depth_tolerance', 1.0)
        self.declare_parameter('refine_tile_size', 32.0)
        self.declare_parameter('yaml_path', '~/ros2_ws/src/stereo_active/config/SM3.yaml')
        self.declare_parameter('tile', 2)
        self.declare_parameter('climp', 8.0)
        self.declare_parameter('window_size', 3)
        self.declare_parameter('stride', 1)
        self.declare_parameter('threshold', 0.7)
        self.declare_parameter('std_thresh', 20)
        self.declare_parameter('radius', 10)
        self.declare_parameter('neighbours', 10)
        self.declare_parameter('crop_image_factor', 0.9)
        self.declare_parameter('save_filename', "correlation_points")
        self.declare_parameter('debug_save_points', False)
        self.declare_parameter('n_images', 10)
        self.declare_parameter('camera_frame_id', 'SM3/left_camera_link')
        self.declare_parameter('zval', 300)
        self.declare_parameter('ekf', False)
        self.declare_parameter('save_images', False)

        self.yaml_file = self.get_parameter('yaml_path').get_parameter_value().string_value
        self.num_images = self.get_parameter('n_images').get_parameter_value().integer_value
        kernel = self.get_parameter('window_size').get_parameter_value().integer_value
        self.get_logger().info(f'Number of images to be captured: {self.num_images} with kernel {kernel}x{kernel}')
        
        # Initialize the InverseTriangulation class
        self.zscan = SpatialCorrelator(yaml_file=self.yaml_file)

        # Services
        self.process_sm3 = self.create_service(Empty, 'process_sm3', self.process_sm3_callback)
        self.laser_process = self.create_client(ProcessFolder, 'laser_acquisition')
        # Initialize the subscribers
        sub_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=5)

        self.passive_pointcloud_subscriber = self.create_subscription(PointCloud2, 'passive/pointcloud', self.z_limits_global, sub_profile)
        if(self.get_parameter('ekf').get_parameter_value().bool_value):
            self.pose_sub = self.create_subscription(PoseStamped, 'pose', self.pose_cb, sub_profile)
        else:
            self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'pose_cov', self.pose_cb, sub_profile)
            
        # Initialize the publisher
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)

        self.perform_correl = False
        self.wait_save = False
        self.process_data = False
        self.pose_msg = None
        self.transformed_xyz = None
        # Construct variables in case disparity point cloud is not available
        self.zmin = -self.get_parameter('zval').get_parameter_value().integer_value
        self.zmax = self.get_parameter('zval').get_parameter_value().integer_value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._init_scan_worker("correlation", "process_data")
    
    def _reset_scan_buffers(self):
        self.perform_correl = False
        self.wait_save = False

    def _dataset_ready(self):
        self.perform_correl = True
        self._submit_scan(self.image_process)

    def pose_cb(self, msg):
        if not self.process_data:
            self.pose_msg = msg

    def save_pose(self):
        pose = self.pose_msg.pose if self.get_parameter("ekf").value else self.pose_msg.pose.pose
        p = pose.position
        q = pose.orientation
        t = self.pose_msg.header.stamp.sec + self.pose_msg.header.stamp.nanosec * 1e-9

        # 1 row with 8 columns: [timestamp, x, y, z, qx, qy, qz, qw]
        return np.array([[t, p.x, p.y, p.z, q.x, q.y, q.z, q.w]], dtype=np.float64)

    
    def image_process(self):
        tile = self.get_parameter('tile').get_parameter_value().integer_value
        climp = self.get_parameter('climp').get_parameter_value().double_value
        n_imgs = self.get_parameter('n_images').get_parameter_value().integer_value

        if self.perform_correl:
            pairs = validate_pairs(self.folder, "laser", n_imgs)
            left_images = []
            right_images = []

            if self.pose_msg is not None:
                data = self.save_pose()
                np.savetxt(os.path.join(self.folder,'laser', 'pose.txt'),data, fmt="%.6f", delimiter=", ")
            if self.transformed_xyz is not None:
                np.savetxt(os.path.join(self.folder,'laser', 'passive_pointcloud_mm.txt'),self.transformed_xyz, fmt="%.6f", delimiter=", ")
        
            for left, right in pairs:
                img_l, img_r = cv2.imread(left, cv2.IMREAD_GRAYSCALE), cv2.imread(right, cv2.IMREAD_GRAYSCALE)
                if img_l is None or img_r is None or img_l.shape != img_r.shape:
                    raise ValueError("Invalid stereo pair")
                left_images.append(img_l)
                right_images.append(img_r)

            t0 = time.time()
            blur = self.get_parameter('laser_blur').value
            self.zscan.convert_images(left_images, right_images, apply_clahe=True, undist=False,
                                      apply_blur=blur > 1, blur=blur, tile=tile, climp=climp)
            del left_images, right_images
            if self.get_parameter('reconstruct_laser').value:
                self.triangulation()
            self.get_logger().info('Correlation process finished: {:.2f} s'.format(time.time()-t0))
            # Worker finally releases the buffers, including failed/empty scans.
            


    def process_sm3_callback(self, request, response):
        self._begin_scan(self.laser_process, ProcessFolder)
        return response

    def triangulation(self):
        start = time.monotonic()
        param = lambda name: self.get_parameter(name).value
        self.zscan.max_block_voxels = param('max_block_voxels')
        coarse, fine = reconstruct_laser(
            self.zscan, ((-100., 600.), (-100., 600.), (self.zmin, self.zmax)),
            kernel=param('window_size'), stride=param('stride'), threshold=param('threshold'),
            texture=param('std_thresh'), crop=param('crop_image_factor'),
            support_radius=param('radius'), min_points=param('surface_min_points'),
            depth_margin=param('depth_band_margin'), split_tolerance=param('split_depth_tolerance'),
            tile_size=param('refine_tile_size'), log=self.get_logger().info)
        points = fine.cpu().numpy()
        self.publish_pointcloud(points)
        if param('debug_save_points'):
            filename = time.strftime('%Y%m%d') + '_' + param('save_filename') + '.txt'
            np.savetxt(os.path.join(self.folder, 'laser', filename), points, fmt='%.6f')
        self.get_logger().info(
            f'Laser reconstruction: {len(coarse)} coarse, {len(fine)} final points '
            f'in {time.monotonic() - start:.2f} s')

    def z_limits_global(self, points):
        if self.process_data:
            return
        
        sm3_frame_id = self.get_parameter('camera_frame_id').value
        T_left = self.zscan.camera_params['left']['t'].cpu().numpy().T[0]  # Obter translação da câmera esquerda
        R_left = self.zscan.camera_params['left']['r'].cpu().numpy()

        # Extract points
        points_generator = pc2.read_points(points, field_names=("x", "y", "z"), skip_nans=True)

        points_list = []
        for p in points_generator:
            points_list.append([p[0], p[1], p[2]])

        points_xyz_cam = np.array(points_list, dtype=np.float32)

        # self.get_logger().info(f'Points XYZ shape: {points_xyz.shape}, dtype: {points_xyz.dtype}')
        if points_xyz_cam.shape[0] == 0:
            self.get_logger().warn('No valid points found in the point cloud after skipping NaNs.')
            return
            
        # Add a fourth dimension (1) for homogeneous coordinates
        points_homogeneous = np.hstack((points_xyz_cam, np.ones((points_xyz_cam.shape[0], 1))))

        # Apply the transformation
        try:
            tf_passive2active = self.tf_buffer.lookup_transform(sm3_frame_id, points.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Transform from {points.header.frame_id} to {sm3_frame_id} not found: {e}")
            return
        T_passive2active = self.do_transform_matrix(tf_passive2active)
        transformed_ph = (T_passive2active @ points_homogeneous.T).T

        # Extract the XYZ coordinates
        self.transformed_xyz_cam = transformed_ph[:, :3] * 1000  # Convert from meters to millimeters

        # transformar os pontos para peça
        self.transformed_xyz = (R_left.T @ (self.transformed_xyz_cam.T - T_left[:, None])).T

        xmin, xmax = -100, 500
        ymin, ymax = -100, 400

        xmask = (self.transformed_xyz[:, 0] >= xmin) & (self.transformed_xyz[:, 0] <= xmax)
        ymask = (self.transformed_xyz[:, 1] >= ymin) & (self.transformed_xyz[:, 1] <= ymax)
        mask = xmask & ymask
        filtered_points = self.transformed_xyz[mask]

        if filtered_points.size == 0:
            return
        # Obtém os limites globais dos pontos
        self.zmin = np.min(filtered_points[:, 2]) - 100  # Consider only Z values
        self.zmax = np.max(filtered_points[:, 2]) + 100  # Consider only Z values
        if abs(self.zmin) + abs(self.zmax) > 1500:
            self.zmax = 1500 - abs(self.zmin)
            # self.get_logger().warning(f'Z limits are too large: zmin={self.zmin}, zmax={self.zmax}. Resetting to default values.')

    def do_transform_matrix(self, msg):
        # Trnasforma as mensagens de PoseStamped e TransformStamped em uma matriz de transformação 4x4

        translation = msg.transform.translation
        rotation = msg.transform.rotation
        parent = msg.header.frame_id
        child = msg.child_frame_id

        tx, ty, tz = translation.x, translation.y, translation.z
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w

        rot = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

        # Create translation vector
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rot
        transformation_matrix[:3, 3] = [tx, ty, tz]

        # self.get_logger().info(f"From {child} to {parent}")

        return transformation_matrix

    def publish_pointcloud(self, points):
        T_left = self.zscan.camera_params['left']['t'].cpu().numpy().T[0]  # Obter translação da câmera esquerda
        R_left = self.zscan.camera_params['left']['r'].cpu().numpy()

        # rotação primiero e translação depois
        points = (R_left @ points.T).T + T_left
        # points = points + T_left

        if points is not None:
            pointcloud_msg = self.convert_to_pointcloud2(points)
            self.pointcloud_publisher.publish(pointcloud_msg)

    def convert_to_pointcloud2(self, points):
        self.frame_id = self.get_parameter('camera_frame_id').value
        # Converte para mensagem PointCloud2
        header = Header()
        header.stamp = self._scan_stamp or self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Corrige a escala dos pontos de metros para milímetros
        points = np.divide(points, 1000.0)
        
        pointcloud_data = b''.join([struct.pack('fff', *p) for p in points])

        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(points),
            data=pointcloud_data,
            is_dense=True
        )

def main(args=None):
    rclpy.init(args=args)
    node = InverseTriangulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()