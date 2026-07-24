from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='', description='Path to RL model'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('action_type', default_value='continuous'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='robot_controller',
            executable='rl_node',
            name='rl_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'camera_topic': LaunchConfiguration('camera_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                'action_type': LaunchConfiguration('action_type'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'inference_rate': 20.0,
                'max_linear_speed': 2.0,
                'max_angular_speed': 4.0,
            }],
        ),
    ])
