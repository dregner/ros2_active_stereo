#!/usr/bin/env python3
import os
import datetime
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Trigger, Empty
import struct
import time
import tf2_ros
from ros2_active_stereo_msgs.srv import ProcessFolder
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header
import tf_transformations

from SpatialCorrelation_pytorch import PyTorchStereoCorrel
import torch
from pipeline_utils import ScanWorkerMixin, stamp_key

class TriangulationNode(ScanWorkerMixin, Node):

    def __init__(self):
        super().__init__('triangulation_node')

        self.bridge = CvBridge()
      

        # Subscribers de tópicos de imagem e camera_info
        self.create_subscription(Image, 'sync/left/phase_map', lambda msg: self.image_callback(msg, 'sync/left/phase_map'), 10)
        self.create_subscription(Image, 'sync/right/phase_map', lambda msg: self.image_callback(msg, 'sync/right/phase_map'), 10)
        self.create_subscription(Image, 'sync/left/modulation_map', lambda msg: self.image_callback(msg, 'sync/left/modulation_map'), 10)
        self.create_subscription(Image, 'sync/right/modulation_map', lambda msg: self.image_callback(msg, 'sync/right/modulation_map'), 10)

        # Publisher de nuvem de pontos
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)

        # Parameters
        self.declare_parameter('yaml_path', '/home/jetson/ros2_ws/src/ros2_active_stereo/params/lab_active.yaml')
        self.declare_parameter('mod_thresh', 30) #cupy 0.07. torch (0-255)
        self.declare_parameter('rad_thresh', 0.05) #threshold for radian difference
        self.declare_parameter('debug_save_points', False)
        self.declare_parameter('save_filename', 'fringe_points')
        self.declare_parameter('camera_frame_id', '/Active/left_camera_link')
        self.declare_parameter('zval', 400)
        self.declare_parameter('neighbours', 5)
        self.declare_parameter('radius', 10)
        self.declare_parameter('ekf', False)
        self.declare_parameter('save_images', False)

        self.zmin = -self.get_parameter('zval').get_parameter_value().integer_value
        self.zmax = self.get_parameter('zval').get_parameter_value().integer_value

        self.images = {
        'sync/left/phase_map': None,
        'sync/right/phase_map': None,
        'sync/left/modulation_map': None,
        'sync/right/modulation_map': None
        }

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.process_triang = False
        self._batches = {}
        # Services
        self.process_sm4 = self.create_service(Empty, 'process_sm4', self.process_sm4_callback)
        self.phase_process = self.create_client(ProcessFolder, 'fringe_acquisition')

        sub_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=5)

        self.passive_pointcloud_subscriber = self.create_subscription(PointCloud2, 'passive/pointcloud', self.z_limits_global, sub_profile)
        if(self.get_parameter('ekf').get_parameter_value().bool_value):
            self.pose_sub = self.create_subscription(PoseStamped, 'pose', self.pose_cb, sub_profile)
        else:
            self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'pose_cov', self.pose_cb, sub_profile)
            
        self.pose_msg = None
        self.transformed_xyz = None

        # Torch class
        self.yaml_file = self.get_parameter('yaml_path').get_parameter_value().string_value
        self.zscan = PyTorchStereoCorrel(yaml_file=self.yaml_file)
        self._init_scan_worker("fringe", "process_triang")
        self.get_logger().info("Node 'triangulation_node' criado")

    def pose_cb(self, msg):
        if not self.process_triang:
            self.pose_msg = msg

    def save_pose(self):
        pose = self.pose_msg.pose if self.get_parameter("ekf").value else self.pose_msg.pose.pose
        p = pose.position
        q = pose.orientation
        t = self.pose_msg.header.stamp.sec + self.pose_msg.header.stamp.nanosec * 1e-9

        # 1 row with 8 columns: [timestamp, x, y, z, qx, qy, qz, qw]
        return np.array([[t, p.x, p.y, p.z, q.x, q.y, q.z, q.w]], dtype=np.float64)

    
    def process_sm4_callback(self, request, response):
        self._begin_scan(self.phase_process, ProcessFolder)
        return response

    def _reset_scan_buffers(self):
        self.images = {key: None for key in self.images}
        self._batches.clear()

    def image_callback(self, msg, image_type):
        with self._scan_lock:
            if not self.process_triang or self._working:
                return
            key = stamp_key(msg.header.stamp)
            if self._expected_stamp is not None and key != self._expected_stamp:
                return
            try:
                batch = self._batches.setdefault(key, {})
                batch[image_type] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Bound unmatched data while waiting for the folder-tagged result.
                while len(self._batches) > 2:
                    del self._batches[min(self._batches)]
                self._dataset_ready()
            except Exception as exc:
                self.get_logger().error(f'Invalid phase map: {exc}')
                self._finish_scan()

    def _dataset_ready(self):
        if self._expected_stamp is None:
            return
        batch = self._batches.get(self._expected_stamp, {})
        if all(key in batch for key in self.images):
            shapes = {img.shape for img in batch.values()}
            if len(shapes) != 1 or len(next(iter(shapes))) != 2:
                self.get_logger().error('Phase and modulation maps must share a 2D shape')
                self._finish_scan()
                return
            self.images = dict(batch)
            self._batches.clear()
            self._submit_scan(self.process_images)

    def process_images(self):
        # TORCH
        left_images = np.asarray([self.images['sync/left/phase_map'], self.images['sync/left/modulation_map']])
        right_images = np.asarray([self.images['sync/right/phase_map'], self.images['sync/right/modulation_map']])
        self.zscan.convert_images(left_images, right_images, apply_clahe=False, undist=True)
        if self.pose_msg is not None:
            data = self.save_pose()
            np.savetxt(os.path.join(self.folder,'fringe', 'pose.txt'),data, fmt="%.6f", delimiter=", ")
        if self.transformed_xyz is not None:
            np.savetxt(os.path.join(self.folder,'fringe', 'passive_pointcloud_mm.txt'),self.transformed_xyz, fmt="%.6f", delimiter=", ")
        self.triangulation()

    def triangulation(self):
        """
            Function to perform spatial correlation
        """
        
        self.get_logger().info("Starting triangulation process.")
        t0 = time.time()
        
        # Get filter points parameters
        mod_tresh = self.get_parameter('mod_thresh').get_parameter_value().integer_value
        rad_thresh = self.get_parameter('rad_thresh').get_parameter_value().double_value
        radius = self.get_parameter('radius').get_parameter_value().integer_value
        min_neighbours = self.get_parameter('neighbours').get_parameter_value().integer_value
        save_points = self.get_parameter('debug_save_points').get_parameter_value().bool_value
        filename = self.get_parameter('save_filename').get_parameter_value().string_value


        GRID_LIMITS = {'x': (-200, 600), 'y': (-200, 600), 'z': (self.zmin, self.zmax)}
        GRID_STEPS_1 = {'xy': 2.0, 'z': 1.0} # first steps of 3d patch
        GRID_STEPS_2 = {'xy': 1.0, 'z': 0.25} # second steps of 3d patch
        
        self.get_logger().info(f'Z range for correlation: ({self.zmin:.2f}, {self.zmax:.2f})')

        self.zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim   =GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                            xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=1, Ky=1, stride=1, Nz_block_voxels=10, method='fringe')
        # filter points based on difference value in radians
        filter_mask = corr_gpu < rad_thresh
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=mod_tresh, method='fringe')
        xyz_filtered_gpu, _,_ = self.zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, min_neighbors=min_neighbours, radius=radius)

        if xyz_filtered_gpu.numel() == 0:
            self.get_logger().warning("No points found")
            return
        
        self.publish_pointcloud(xyz_filtered_gpu.cpu().numpy())
        
        self.get_logger().info(f'1st Triangulation completed in {time.time() - t0:.2f} seconds. Total points: {xyz_filtered_gpu.shape[0]}')
        t0 = time.time()
        # self.publish_pointcloud(xyz_filtered_gpu.cpu().numpy())
        
        # Find first 3D bounds to refined process               
        xlim = torch.min(xyz_filtered_gpu[:, 0]), torch.max(xyz_filtered_gpu[:, 0])
        ylim = torch.min(xyz_filtered_gpu[:, 1]), torch.max(xyz_filtered_gpu[:, 1])
        zlim = torch.min(xyz_filtered_gpu[:, 2]), torch.max(xyz_filtered_gpu[:, 2])

        # self.get_logger().info(f'3D bounds for refined triangulation: X: {xlim}, Y: {ylim}, Z: {zlim}')
        if zlim[0] == zlim[1]:
            self.get_logger().info("Z are same")
            zlim = (zlim[0] - 10, zlim[1] + 10)

        # Construct second 3d points
        self.zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, 
                            xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=1, Ky=1, stride=1, Nz_block_voxels=60, method='fringe')

        
        filter_mask = corr_gpu < rad_thresh
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=mod_tresh, method='fringe')
        final_xyz_gpu, _,_ = self.zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, min_neighbors=min_neighbours, radius=radius)


        # Publicar os pontos refinados
        self.get_logger().info(f'2nd Triangulation completed in {time.time() - t0:.2f} seconds. Total points: {final_xyz_gpu.shape[0]}')
        self.publish_pointcloud(final_xyz_gpu.cpu().numpy())

        if save_points:
            # Salvar os pontos refinados em um arquivo .txt
            np.savetxt(os.path.join(self.folder,'fringe','{}_{}.txt'.format(time.strftime("%Y%m%d"), filename)), final_xyz_gpu.cpu().numpy(), fmt='%.6f')

   
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

    def z_limits_global(self, points):
        if self.process_triang:
            return

        sm4_frame_id = self.get_parameter('camera_frame_id').value
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
            tf_sm2_sm4 = self.tf_buffer.lookup_transform(sm4_frame_id, points.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Transform from {points.header.frame_id} to {sm4_frame_id} not found: {e}")
            return
        
        T_sm2_sm4 = self.do_transform_matrix(tf_sm2_sm4)
        transformed_ph = (T_sm2_sm4 @ points_homogeneous.T).T

        # Extract the XYZ coordinates
        self.transformed_xyz_cam = transformed_ph[:, :3] * 1000  # Convert from meters to millimeters

        # transformar os pontos para peça
        self.transformed_xyz = ((R_left.T @ self.transformed_xyz_cam.T) - (R_left.T @ T_left[:, None])).T

        xmin, xmax = -100, 500
        ymin, ymax = -100, 400

        xmask = (self.transformed_xyz[:, 0] >= xmin) & (self.transformed_xyz[:, 0] <= xmax)
        ymask = (self.transformed_xyz[:, 1] >= ymin) & (self.transformed_xyz[:, 1] <= ymax)
        mask = xmask & ymask
        filtered_points = self.transformed_xyz[mask]

        if filtered_points.size == 0:
            return
        # Obtém os limites globais dos pontos
        self.zmin = np.min(filtered_points[:, 2]) - 100 # Consider only Z values
        self.zmax = np.max(filtered_points[:, 2]) + 100# Consider only Z values
        if abs(self.zmin) + abs(self.zmax) > 1000:
            self.zmax = 1000 - abs(self.zmin)
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

def main(args=None):
    rclpy.init(args=args)
    node = TriangulationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
