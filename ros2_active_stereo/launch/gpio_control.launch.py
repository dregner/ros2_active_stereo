from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='SM3',
            description='Namespace for node'
        ),
        DeclareLaunchArgument(
            'stepping_mode',
            default_value='full',
            description='Stepper motor mode (full or half)'
        ),
        DeclareLaunchArgument(
            'step_delay',
            default_value='3500',
            description='Delay between steps in ms'
        ),
        DeclareLaunchArgument(  
            'steps_per_rev',
            default_value='2048',
            description='Steps per revolution'
        ),
        DeclareLaunchArgument(  
            'motor_angle_topic',
            default_value='motor/angle',
            description='Topic of stepper motor angle'
        ),

        Node(
            package='ros2_active_stereo',
            executable='gpio_control',  
            namespace=LaunchConfiguration('namespace'),
            name='gpio_control_node',
            output='screen',
            parameters=[
                {'stepping_mode': LaunchConfiguration('stepping_mode')},
                {'delay': LaunchConfiguration('step_delay')},
                {'steps_per_revolution': LaunchConfiguration('steps_per_rev')}
            ],
            remappings=[
                ('motor/angle', LaunchConfiguration('motor_angle_topic')),
            ]
        ),

    ])
