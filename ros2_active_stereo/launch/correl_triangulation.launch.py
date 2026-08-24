from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
        return LaunchDescription([
            DeclareLaunchArgument(
                'namespace',
                default_value='Active',
                description='Namespace'
            ),
            DeclareLaunchArgument(
                'point_cloud',
                default_value='SM3/pointcloud',
                description='Point cloud topic'
            ),
            DeclareLaunchArgument(
                'disparity_pointcloud',
                default_value='disparity/pointcloud',
                description='Disparity point cloud topic'
            ),
            DeclareLaunchArgument(
                'neighbours',
                default_value='15',
                description='num_splits'
            ),
            DeclareLaunchArgument(
                'radius',
                default_value='5',
                description='num_splits'
            ),
            DeclareLaunchArgument(
                'n_images',
                default_value='20',
                description='Number of images to acquire'
            ),
            DeclareLaunchArgument(
                'window_size',
                default_value='3',
                description='Number of images to acquire'
            ),
            DeclareLaunchArgument(
                'yaml_path',
                default_value=PathJoinSubstitution([FindPackageShare('stereo_active'), 'config','SM3.yaml']),
                description='Number of images to acquire'
            ),
            DeclareLaunchArgument(
                'camera_frame_id',
                default_value='Active/left_camera_link',
                description='Camera frame ID'
            ),
            DeclareLaunchArgument(
                'correl_thresh',
                default_value='0.7',
                description='Correlation threshold'
            ),
            DeclareLaunchArgument(
                'std_thresh',
                default_value='20',
                description='Standard deviation threshold'
            ),
            DeclareLaunchArgument(
                'zval',
                default_value='400',
                description='Z value for triangulation'
            ),
            Node(
                package='ros2_active_stereo',
                executable='triangulation_correl_node.py',
                name='inv_correlation_node',
                namespace=LaunchConfiguration('namespace'),
                output='screen',
                parameters=[
                    {'num_images': LaunchConfiguration('n_images'),
                     'yaml_path': LaunchConfiguration('yaml_path'),
                     'window_size': LaunchConfiguration('window_size'),
                     'camera_frame_id': LaunchConfiguration('camera_frame_id'),
                     'threshold': LaunchConfiguration('correl_thresh'),
                     'std_thresh': LaunchConfiguration('std_thresh'),
                     'neightbours': LaunchConfiguration('neighbours'),
                     'radius': LaunchConfiguration('radius'),
                     'zval': LaunchConfiguration('zval')}
                ],
                remappings=[
                    ('pointcloud', LaunchConfiguration('point_cloud')),
                    ('disparity/pointcloud', LaunchConfiguration('disparity_pointcloud'))
                ]
            )
        ])