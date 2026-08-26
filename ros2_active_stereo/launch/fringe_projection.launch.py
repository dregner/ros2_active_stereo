from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
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
                'hz',
                default_value='10',
                description='Frequency'
            ),
        DeclareLaunchArgument(
                'px_f',
                default_value='64',
                description='px_f'
            ),
        DeclareLaunchArgument(
                'steps',
                default_value='8',
                description='steps'
            ),
        DeclareLaunchArgument(
                'index',
                default_value='0',
                description='index'
            ),
        DeclareLaunchArgument(
                'yaml_path',
                default_value='/home/jetson/ros2_ws/src/ros2_fringe_projection/params/SM4.yaml',
                description='Caminho do arquivo YAML com parâmetros da calibração'
            ),



        Node(
        package='ros2_active_stereo',
        executable='phase_processing_node.py',
        name='phase_processing_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters = [
            {'hz': LaunchConfiguration('hz')},
            {'px_f': LaunchConfiguration('px_f')},
            {'steps': LaunchConfiguration('steps')},
            {'index': LaunchConfiguration('index')},
            ],
            remappings= [
                ('left/image', 'left/image_raw'), 
                ('right/image', 'right/image_raw')]
        ),
    ])