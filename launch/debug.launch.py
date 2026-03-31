from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

# def launch_setup(context, *args, **kwargs):
#     composable_nodes = [
#         ComposableNode(
#             package='ros2_active_stereo',
#             plugin='ImageProjectNode',
#             name='camera_node',
#             parameters=[{
#                 'pixel_per_fringe': 128,
#                 'fringe_steps': 4,
#                 'image_color': 'blue',
#             }],
#             extra_arguments=[{'use_intra_process_comms': False}],
#         ),
#         ComposableNode(
#             package='ros2_active_stereo',
#             plugin='StereoAcquisitionNode',
#             name='stereo_acquisition_node',
#             remappings=[
#                 ('left/image_raw', 'camera/left/image_raw'),
#                 ('right/image_raw', 'camera/right/image_raw'),
#             ],
#             extra_arguments=[{'use_intra_process_comms': False}],
#         ),

#     ]

#     container = ComposableNodeContainer(
#         name='cam_sync_container',
#         package='rclcpp_components',
#         namespace='',
#         executable='component_container',
#         composable_node_descriptions=composable_nodes,
#         output='screen',
#     )


#     return [container]

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='ros2_active_stereo',
            executable='image_project_node',
            name='image_project_node',
            parameters=[{
                'pixel_per_fringe': 128,
                'fringe_steps': 4,
                'image_color': 'blue',
            }],
            output='screen',
        ),
        Node(
            package='ros2_active_stereo',
            executable='stereo_acquisition_node',
            name='stereo_acquisition_node',
            remappings=[
                ('left/image_raw', 'camera/left/image_raw'),
                ('right/image_raw', 'camera/right/image_raw'),
            ],
            output='screen',
        )
    ])