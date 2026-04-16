from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
                'namespace',
                default_value='Active',
                description='Namespace'
            ),
        DeclareLaunchArgument(
                'mod_tresh',
                default_value='30',
                description='num_splits'
            ),
        DeclareLaunchArgument(
                'rad_tresh',
                default_value='0.05',
                description='num_splits'
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
                'camera_frame_id',
                default_value='Active/left_camera_link',
                description='Camera frame ID'
            ),
        DeclareLaunchArgument(
                'yaml_path',
                default_value='/home/jetson/ros2_ws/src/ros2_active_stereo/ros2_active_stereo/config/lab_active.yaml',
                description='Path to the YAML file containing camera parameters'
            ),
        DeclareLaunchArgument(
                'absolute_image',
                default_value='abs_phi',
                description='Topic name for the absolute image'
            ),
        DeclareLaunchArgument(
                'modulation_image',
                default_value='mask',
                description='Topic name for the modulation image'
            ),
        DeclareLaunchArgument(
                'disparity_pointcloud',
                default_value='disparity/pointcloud',
                description='Topic name for the disparity point cloud'
            ),
            DeclareLaunchArgument(
                'triangulated_pointcloud',
                default_value='triangulated/pointcloud',
                description='Topic name for the triangulated point cloud'
            ),

        Node(
                package='ros2_active_stereo',
                executable='triangulation_node.py',
                name='structured_light_inverse_triang_node',
                namespace=LaunchConfiguration('namespace'),
                parameters=[{
                    'yaml_path': LaunchConfiguration('yaml_path'),
                    'mod_thresh': LaunchConfiguration('mod_tresh'),
                    'rad_tresh': LaunchConfiguration('rad_tresh'),
                    'camera_frame_id': LaunchConfiguration('camera_frame_id'),
                    'neightbours': LaunchConfiguration('neighbours'),
                    'radius': LaunchConfiguration('radius'),
                    'zval': 300,
                }],
                remappings=[
                    ('abs_phi_left', 'sync/left/phase_map'),
                    ('abs_phi_right', 'sync/right/phase_map'),
                    ('mask_left', 'sync/left/modulation_map'),
                    ('mask_right', 'sync/right/modulation_map'),
                    ('passive/pointcloud', LaunchConfiguration('disparity_pointcloud')),
                    ('pointcloud', LaunchConfiguration('triangulated_pointcloud'))
                ]
            ),
    ])