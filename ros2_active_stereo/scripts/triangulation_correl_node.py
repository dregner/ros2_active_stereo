#!/usr/bin/env python3
import cv2
import numpy as np
import time
import struct
import torch
import gc

from SpatialCorrelation_pytorch import PyTorchStereoCorrel as SpatialCorrelator

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Int16
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import tf_transformations

# retirar metodos que foram passados para cpp
# importar as imagens direto da ram (salvas pelo inv_correlation_node.cpp)
# handshake para importar as imagens


class InverseTriangulationNode(Node):
    def __init__(self):
        super().__init__('inverse_correlation_node')
        self.get_logger().info('Inverse Correlation Node has been started.')

        # Parameters declaration
        self.declare_parameter('yaml_path', '~/ros2_ws/src/stereo_active/config/SM3.yaml')
        self.declare_parameter('tile', 1)
        self.declare_parameter('climp', 5.0)
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

        self.yaml_file = self.get_parameter('yaml_path').get_parameter_value().string_value
        self.num_images = self.get_parameter('n_images').get_parameter_value().integer_value
        kernel = self.get_parameter('window_size').get_parameter_value().integer_value
        self.get_logger().info(f'Number of images to be captured: {self.num_images} with kernel {kernel}x{kernel}')
        
        # Initialize the InverseTriangulation class
        self.zscan = SpatialCorrelator(yaml_file=self.yaml_file)

        self.left_images = []
        self.right_images = []

        # Initialize the subscribers
        sub_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=5)

        self.passive_pcl_sub = self.create_subscription(PointCloud2, 'disparity/pointcloud', self.z_limits_global, sub_profile)
        self.handshake_images_sub = self.create_subscription(Int16, 'handshake_images', self.handshake_images_cb, 10)
        
        # Initialize the publisher
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)

        self.perform_correl = False

        # Construct variables in case disparity point cloud is not available
        self.zmin = -self.get_parameter('zval').get_parameter_value().integer_value
        self.zmax = self.get_parameter('zval').get_parameter_value().integer_value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def handshake_images_cb(self, msg):
        #importa as mensagens da memoria ram

        #self.get_logger().info(f"Handshake do c++ concluido, num_images= {msg.data}")
        self.num_images = abs(msg.data)         # verifica o sinal do numero de img recebido para definir se realiza a correlacao
        self.perform_correl = msg.data > 0
        base_path = '/tmp/rrp_stereo'

        self.left_images = []
        self.right_images = []

        for n in range(1, self.num_images + 1):
            left_img_path = f"{base_path}/left/L{n:02d}.png"
            right_img_path = f"{base_path}/right/R{n:02d}.png"

            left_image = cv2.imread(left_img_path,cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(right_img_path,cv2.IMREAD_GRAYSCALE)

            if left_image is None or right_image is None:
                self.get_logger().error(f'Erro ao carregar a imagem {n}')
                return

            self.left_images.append(left_image)
            self.right_images.append(right_image)

        self.image_process()
        
    def image_process(self):
        tile = self.get_parameter('tile').get_parameter_value().integer_value
        climp = self.get_parameter('climp').get_parameter_value().double_value

        if self.perform_correl:
            t0 = time.time()
            self.zscan.convert_images(left_imgs_cpu=self.left_images, right_imgs_cpu=self.right_images, apply_clahe=True, undist=True, tile=tile, climp=climp)
            self.get_logger().info('Images converted: {:.2f} s'.format(time.time()-t0))
            self.triangulation()
            self.get_logger().info('Correlation process finished: {:.2f} s'.format(time.time()-t0))
            self.perform_correl = False
            self.left_images.clear()
            self.right_images.clear()
            
            # --- Limpando a memoria
            # 1. Cortamos as referências dos tensores dentro da classe PyTorch
            self.zscan.left_images = None
            self.zscan.right_images = None
            self.zscan.grid = None
            self.zscan.x_vals = None
            self.zscan.y_vals = None
            self.zscan.z_vals = None

            # 2. Forçamos o Python a reconhecer que os objetos estão órfãos
            gc.collect()
            
            # 3. Agora sim, esvaziamos a VRAM da Jetson
            torch.cuda.empty_cache()
            self.get_logger().info('Memoria limpa')
            # -------------------------------
            

    def triangulation(self):
        """
            Function to perform spatial correlation
        """

        self.get_logger().info("Starting triangulation process.")
        t0 = time.time()
        # Get filter points parameters
        std_thresh = self.get_parameter('std_thresh').get_parameter_value().integer_value
        correl_thresh = self.get_parameter('threshold').get_parameter_value().double_value
        win_size = self.get_parameter('window_size').get_parameter_value().integer_value
        stride = self.get_parameter('stride').get_parameter_value().integer_value

        radius = self.get_parameter('radius').get_parameter_value().integer_value
        min_neighbours = self.get_parameter('neighbours').get_parameter_value().integer_value
        save_points = self.get_parameter('debug_save_points').get_parameter_value().bool_value
        filename = self.get_parameter('save_filename').get_parameter_value().string_value
        crop_factor = self.get_parameter('crop_image_factor').get_parameter_value().double_value

        self.get_logger().info(f'Threshold: {correl_thresh*100} %, Win: {win_size}x{win_size}x{self.num_images}')

        GRID_LIMITS = {'x': (-100, 600), 'y': (-100, 600), 'z': (self.zmin, self.zmax)}
        GRID_STEPS_1 = {'xy': 2.0, 'z': 2.0} # first steps of 3d patch
        GRID_STEPS_2 = {'xy': 1.0, 'z': 0.1} # second steps of 3d patch
        # GRID_STEPS_3= {'xy': 1.0, 'z': 0.01} # second steps of 3d patch

        self.get_logger().info(f'Z range for correlation: ({self.zmin:.2f}, {self.zmax:.2f})')

        self.zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                            xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=win_size, Ky=win_size, stride=stride, Nz_block_voxels=10, method='correl')

        # filter points based on difference value in radians
        filter_mask = corr_gpu > 0.6
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu, crop_factor=crop_factor)
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=std_thresh, method='correl')
        # clean points based on neighbours
        final_xyz_gpu, _,_ = self.zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, min_neighbors=min_neighbours, radius=radius)
        
        if final_xyz_gpu.numel() == 0:
            self.get_logger().warning("No points found")
            return

        self.get_logger().info(f'1st Triangulation completed in {time.time() - t0:.2f} seconds. Total points: {final_xyz_gpu.shape[0]}')
        self.publish_pointcloud(final_xyz_gpu.cpu().numpy())
        # Find first 3D bounds to refined process               
        xlim = torch.min(final_xyz_gpu[:, 0]), torch.max(final_xyz_gpu[:, 0])
        ylim = torch.min(final_xyz_gpu[:, 1]), torch.max(final_xyz_gpu[:, 1])
        zlim = torch.min(final_xyz_gpu[:, 2]), torch.max(final_xyz_gpu[:, 2])

        if zlim[0] == zlim[1]:
            self.get_logger().info("Z are same")
            zlim[0] = zlim[0] - 5
            zlim[1] = zlim[1] + 5

        # Construct second 3d points
        self.zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, 
                            xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=win_size, Ky=win_size, stride=stride, Nz_block_voxels=5, method='correl') 

        
        filter_mask = corr_gpu > correl_thresh
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.mask_uv_points(xyz_filtered_gpu, corr_filtered_gpu, crop_factor=crop_factor)
        xyz_filtered_gpu, corr_filtered_gpu, _ = self.zscan.std_mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=std_thresh, method='correl')
        final_xyz_gpu, _,_ = self.zscan.euclidean_filter(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, min_neighbors=min_neighbours*2, radius=radius/2)

        self.get_logger().info(f'2nd Triangulation completed in {time.time() - t0:.2f} seconds. Total points: {final_xyz_gpu.shape[0]}')
        self.publish_pointcloud(final_xyz_gpu.cpu().numpy())


        if save_points:
            np.savetxt('{}_{}.txt'.format(time.strftime("%Y%m%d"), filename), final_xyz_gpu.cpu().numpy(), fmt='%.6f')
        
        del xyz_gpu, corr_gpu, xyz_filtered_gpu, corr_filtered_gpu, final_xyz_gpu # limpando memoria
            
    def z_limits_global(self, points):
        
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
        transformed_xyz_cam = transformed_ph[:, :3] * 1000  # Convert from meters to millimeters

        # transformar os pontos para peça
        transformed_xyz = ((R_left.T @ transformed_xyz_cam.T) - T_left[:, None]).T

        xmin, xmax = -100, 500
        ymin, ymax = -100, 400

        xmask = (transformed_xyz[:, 0] >= xmin) & (transformed_xyz[:, 0] <= xmax)
        ymask = (transformed_xyz[:, 1] >= ymin) & (transformed_xyz[:, 1] <= ymax)
        mask = xmask & ymask
        filtered_points = transformed_xyz[mask]

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
        header.stamp = self.get_clock().now().to_msg()
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