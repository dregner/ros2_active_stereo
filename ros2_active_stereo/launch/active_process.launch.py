from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    # config_file = os.path.join( get_package_share_directory('ros2_active_stereo'), 'config', 'stereo_active_params.yaml' )
    config_file = f'/home/{os.environ["USER"]}/ros2_ws/src/ros2_active_stereo/ros2_active_stereo/config/stereo_active_params.yaml'
    return LaunchDescription([

        DeclareLaunchArgument('passive_pointcloud', default_value='/Passive/disparity/pointcloud', description='Passive pointcloud data'),
        DeclareLaunchArgument('namespace', default_value='Active', description='Nodes namespace'),
        DeclareLaunchArgument('ekf', default_value='False', description='Uses EKF pose data?'),
        DeclareLaunchArgument('debug_show', default_value='True', description='Show debug images'),
        DeclareLaunchArgument('save_images', default_value='True', description='Save iamges?'),
        
    

        Node(
                package='ros2_active_stereo',
                executable='external_acquire_node.py',
                name='acquisition_node',
                namespace=LaunchConfiguration('namespace'),
                parameters=[
                    config_file,
                    {'debug_show': LaunchConfiguration('debug_show'),
                }]
        ),

        Node(
                package='ros2_active_stereo',
                executable='triangulation_correl_node.py',
                name='triangulation_correl_node',
                namespace=LaunchConfiguration('namespace'),
                output='screen',
                parameters=[
                    config_file,
                    {
                    'save_images': LaunchConfiguration('save_images'),
                    'ekf': LaunchConfiguration('ekf')
                    }],
                remappings=[
                    ('passive/pointcloud', LaunchConfiguration('passive_pointcloud')),
                    ('pointcloud', 'SM3/pointcloud'),
                    ('pose', '/mavros/local_position/pose'),
                    ('pose_cov', '/Passive/slam/pose_cov')
                ]
            ),

        Node(
                package='ros2_active_stereo',
                executable='triangulation_fringe_node.py',
                name='triangulation_fringe_node',
                namespace=LaunchConfiguration('namespace'),
                parameters=[
                    config_file,
                    {
                    'save_images': LaunchConfiguration('save_images'),
                    'ekf': LaunchConfiguration('ekf')
                    }],
                remappings=[
                    ('passive/pointcloud', LaunchConfiguration('passive_pointcloud')),
                    ('pointcloud', 'SM4/pointcloud'),
                    ('pose', '/mavros/local_position/pose'),
                    ('pose_cov', '/Passive/slam/pose_cov')
                ]
            ),
    ])