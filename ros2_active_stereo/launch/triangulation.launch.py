from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
                'namespace',
                default_value='SM4',
                description='Namespace'
            ),
        DeclareLaunchArgument(
                'mod_tresh',
                default_value='50',
                description='num_splits'
            ),
        DeclareLaunchArgument(
                'rad_tresh',
                default_value='0.06',
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
                default_value='/SM4/left_camera_link',
                description='Camera frame ID'
            ),
        DeclareLaunchArgument(
                'yaml_path',
                default_value='/home/jetson/ros2_ws/src/ros2_active_stereo/ros2_active_stereo/params/lab_active.yaml',
                description='Path to the YAML file containing camera parameters'
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
                }]
            ),
    ])