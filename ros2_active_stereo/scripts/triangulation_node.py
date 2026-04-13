#!/usr/bin/env python3
import numpy as np
import cupy as cp
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Empty
from std_msgs.msg import String
import struct
import time
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import tf_transformations

from ros2_active_stereo.ros2_active_stereo.scripts.SpatialCorrelation_pytorch import PyTorchStereoCorrel # pyright: ignore[reportMissingImports]
import torch

class TriangulationNode(Node):

    def __init__(self):
        super().__init__('triangulation_node')

        self.bridge = CvBridge()

        self.get_logger().info("Node 'triangulation_node' criado, esperando transições do ciclo de vida...")

        self.state_pub = self.create_publisher(String, 'state_triangulation_node', 10)
        self.current_state = 'UNCONFIGURED'
        self.set_state(self.current_state)

        # Timer que publica o estado a cada 1 segundo
        self.state_timer = self.create_timer(0.5, self.publish_state_periodically)

        self.zscan = None
        # Construct variables in case disparity point cloud is not available
        self.zmin = -200
        self.zmax = 200

        # Configuring the node
        self.on_configure()

    def set_state(self, state_name: str):
        self.current_state = state_name
        msg = String()
        msg.data = state_name
        self.state_pub.publish(msg)
        self.get_logger().info(f"[Estado] {state_name}")

    def publish_state_periodically(self):
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    def on_configure(self):
        try: 
            self.get_logger().info('triangulation_node is configuring...')
            self.current_state = 'CONFIGURING'
            self.set_state(self.current_state)

            # Subscribers de tópicos de imagem e camera_info
            self.create_subscription(Image, 'abs_phi_left', lambda msg: self.image_callback(msg, 'abs_phi_left'), 10)
            self.create_subscription(Image, 'abs_phi_right', lambda msg: self.image_callback(msg, 'abs_phi_right'), 10)
            self.create_subscription(Image, 'mask_left', lambda msg: self.image_callback(msg, 'mask_left'), 10)
            self.create_subscription(Image, 'mask_right', lambda msg: self.image_callback(msg, 'mask_right'), 10)

            # Publisher de nuvem de pontos
            self.pointcloud_publisher = self.create_publisher(PointCloud2, 'pointcloud', 10)

            # Parameters
            self.declare_parameter('yaml_path', '/home/jetson/ros2_ws/src/ros2_fringe_projection/params/SM4.yaml')
            self.declare_parameter('mod_thresh', 50) #cupy 0.07. torch (0-255)
            self.declare_parameter('rad_tresh', 0.06) #threshold for radian difference
            self.declare_parameter('debug_save_points', False)
            self.declare_parameter('save_filename', 'fringe_points')
            self.declare_parameter('camera_frame_id', '/Active/left_camera_link')
            # KDTree parameters
            self.declare_parameter('neighbours', 15)
            self.declare_parameter('radius', 12)

            self.images = {
            'abs_phi_left': None,
            'abs_phi_right': None,
            'mask_left': None,
            'mask_right': None
            }

            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            # Services
            self.process_phase = self.create_client(Empty, 'phase_process')
            while not self.process_phase.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Service not available, waiting again ...')

            self.process_sm4 = self.create_service(Empty, 'process_sm4', self.process_sm4_callback)

        except Exception as e:
            self.get_logger().info(f'Error during configuration: {e}')
            self.current_state = 'ERROR IN CONFIGURATION'
            self.set_state(self.current_state)

    def on_activate(self):
        try:
            self.get_logger().info('triangulation_node  is activating...')

            self.passive_pointcloud_subscriber = self.create_subscription(PointCloud2, '/Passive/disparity/pointcloud', self.z_limits_global, 10)

            self.images = {
            'abs_phi_left': None,
            'abs_phi_right': None,
            'mask_left': None,
            'mask_right': None
            }
            self.mod_thresh = self.get_parameter('mod_thresh').value
            self.yaml_file = self.get_parameter('yaml_path').get_parameter_value().string_value
            filename = self.get_parameter('save_filename').get_parameter_value().string_value
            # Torch class
            self.zscan = PyTorchStereoCorrel(yaml_file=self.yaml_file)
            self.get_logger().info('triangulation_node activated successfully.')

        except Exception as e:
            self.get_logger().info(f'Error during activation: {e}')
            self.current_state = 'ERROR IN ACTIVATION'
            self.set_state(self.current_state)

    def _phase_process(self):
        request = Empty.Request()
        future = self.process_phase.call_async(request)
        future.add_done_callback(self._phase_callback)
    
    def _phase_callback(self, future):
        try:
            future.result()
            self.on_activate()
            self.get_logger().info('Serviço de fase executado com sucesso')
        except Exception as e:
            self.get_logger().error(f'Error ao chamar o serviço de fase: {e}')

    def process_sm4_callback(self, request, response):
        self.get_logger().info('Inicializando sistema de medição de projeção de franjas')
        self.current_state = 'ACTIVATING'
        self.set_state(self.current_state)
        self.process_triang = True
        self._phase_process()
        return response
    
    def image_callback(self, msg, image_type):

        # Atualiza o dicionário com a imagem/máscara correspondente
        self.images[image_type] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if self.zscan is not None:
            self.check_and_process_images()
        else:
            self.get_logger().warn("zscan is not initialized yet. Waiting for camera info.")

    def check_and_process_images(self):

        # Verifica se todas as imagens foram recebidas
        if all(image is not None for image in self.images.values()):
            if self.process_triang:
                self.process_images()

    def process_images(self):
        # TORCH
        left_images = np.asarray([self.images['abs_phi_left'], self.images['mask_left']])
        right_images = np.asarray([self.images['abs_phi_right'], self.images['mask_right']])
        self.zscan.convert_images(left_images, right_images, apply_clahe=False, undist=True)

        self.triangulation()

    def triangulation(self):
        self.get_logger().info(f'Z range: ({self.zmin:.2f}, {self.zmax:.2f})')

        self.get_logger().info("Starting triangulation process.")
        self.current_state = 'ACTIVATED'
        self.set_state(self.current_state)
        t0 = time.time()
        
        # Get filter points parameters
        mod_tresh = self.get_parameter('mod_thresh').value
        rad_tresh = self.get_parameter('rad_tresh').value
        radius = self.get_parameter('radius').value
        min_neighbors = self.get_parameter('neighbours').value
        save_points = self.get_parameter('debug_save_points').value
        filename = self.get_parameter('save_filename').get_parameter_value().string_value


        GRID_LIMITS = {'x': (-100, 500), 'y': (-100, 400), 'z': (self.zmin, self.zmax)}
        GRID_STEPS_1 = {'xy': 2.0, 'z': 2} # first steps of 3d patch
        GRID_STEPS_2 = {'xy': 2.0, 'z': 1} # second steps of 3d patch

        self.zscan.points3d(x_lim=GRID_LIMITS['x'], y_lim=GRID_LIMITS['y'], z_lim=GRID_LIMITS['z'],
                            xy_step=GRID_STEPS_1['xy'], z_step=GRID_STEPS_1['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=1, Ky=1, stride=1, Nz_block_voxels=20, method='fringe')

        # filter points based on difference value in radians
        filter_mask = corr_gpu < rad_tresh
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu = self.zscan.mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=mod_tresh, method='fringe')

        # clean points based on neighbours
        # final_xyz_gpu, _ = self.zscan.filter_sparse_points( xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu,min_neighbors=min_neighbors, radius=radius)
        
        if xyz_filtered_gpu.numel() == 0:
            self.get_logger().warning("No points found")
            return
        
        # Find first 3D bounds to refined process               
        xlim = torch.min(xyz_filtered_gpu[:, 0]), torch.max(xyz_filtered_gpu[:, 0])
        ylim = torch.min(xyz_filtered_gpu[:, 1]), torch.max(xyz_filtered_gpu[:, 1])
        zlim = torch.min(xyz_filtered_gpu[:, 2]), torch.max(xyz_filtered_gpu[:, 2])
        
        if zlim[0] == zlim[1]:
            self.get_logger().info("Z are same")
            zlim[0] = zlim[0] - 5
            zlim[1] = zlim[1] + 5

        # Construct second 3d points
        self.zscan.points3d(x_lim=xlim, y_lim=ylim, z_lim=zlim, 
                            xy_step=GRID_STEPS_2['xy'], z_step=GRID_STEPS_2['z'])
                        
        xyz_gpu, corr_gpu, _ = self.zscan.process_segmented_z(Kx=1, Ky=1, stride=1, Nz_block_voxels=5, method='fringe')

        
        filter_mask = corr_gpu < rad_tresh
        xyz_filtered_gpu = xyz_gpu[filter_mask]
        corr_filtered_gpu = corr_gpu[filter_mask]
        xyz_filtered_gpu, corr_filtered_gpu = self.zscan.mask_points(xyz_filtered_gpu, corr_filtered_gpu, bounds=mod_tresh, method='fringe')
        # final_xyz_gpu, _ = self.zscan.filter_sparse_points(xyz_gpu=xyz_filtered_gpu, corr_gpu=corr_filtered_gpu, min_neighbors=min_neighbors+10, radius=radius-5)
        self.current_state = 'FINALIZED'
        self.set_state(self.current_state)

        if save_points:
            # Salvar os pontos refinados em um arquivo .txt
            self.save_pointcloud(points=xyz_filtered_gpu.cpu().numpy(), filename=filename)

        # Publicar os pontos refinados
        self.publish_pointcloud(xyz_filtered_gpu.cpu().numpy())
        self.process_triang = False
   
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
    
    def save_pointcloud(self, points, filename):
        # Salva os pontos em um arquivo .txt
        np.savetxt('{}.txt'.format(filename), points, fmt='%.6f', delimiter=' ')
        self.get_logger().info(f"Point cloud saved to {filename}")

    def z_limits_global(self, points):

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
        tf_sm2_sm4 = self.tf_buffer.lookup_transform(sm4_frame_id, points.header.frame_id, rclpy.time.Time())
        T_sm2_sm4 = self.do_transform_matrix(tf_sm2_sm4)
        transformed_ph = (T_sm2_sm4 @ points_homogeneous.T).T

        # Extract the XYZ coordinates
        transformed_xyz_cam = transformed_ph[:, :3] * 1000  # Convert from meters to millimeters

        # transformar os pontos para peça
        # transformed_xyz = ((R_left.T @ transformed_xyz_cam.T) - T_left[:, None]).T
        transformed_xyz = ((R_left.T @ transformed_xyz_cam.T) - (R_left.T @ T_left[:, None])).T

        # orientacao = R_left.T # matriz de rotação (3x3)
        # translacao = R_left.T @ T_left # vetor de translação (3x1)

        # matriz_homogenea = np.eye(4)
        # matriz_homogenea[:3, :3] = orientacao
        # matriz_homogenea[:3, 3] = translacao

        # np.set_printoptions(precision=4, suppress=True)
        # self.get_logger().info(f'\nmatriz homogenea:\n{matriz_homogenea}')

        xmin, xmax = -100, 500
        ymin, ymax = -100, 400

        xmask = (transformed_xyz[:, 0] >= xmin) & (transformed_xyz[:, 0] <= xmax)
        ymask = (transformed_xyz[:, 1] >= ymin) & (transformed_xyz[:, 1] <= ymax)
        mask = xmask & ymask
        filtered_points = transformed_xyz[mask]

        # Obtém os limites globais dos pontos
        self.zmin = np.min(filtered_points[:, 2])  # Consider only Z values
        self.zmax = np.max(filtered_points[:, 2])  # Consider only Z values
        # self.get_logger().info(f'Z range: ({self.zmin:.2f}, {self.zmax:.2f})')

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
