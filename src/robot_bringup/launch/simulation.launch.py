import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_robot_description = get_package_share_directory('robot_description')
    pkg_robot_bringup = get_package_share_directory('robot_bringup')
    pkg_robot_sim = get_package_share_directory('robot_sim')
    pkg_robot_controller = get_package_share_directory('robot_controller')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = LaunchConfiguration('world_file', default='obstacle_course.world')
    model_path = LaunchConfiguration('model_path', default='')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_rl = LaunchConfiguration('use_rl', default='true')
    rl_config = LaunchConfiguration('rl_config', default='')
    camera_topic = LaunchConfiguration('camera_topic', default='/camera/image_raw')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic', default='/cmd_vel')
    x_pos = LaunchConfiguration('x_pos', default='0.0')
    y_pos = LaunchConfiguration('y_pos', default='0.0')
    z_pos = LaunchConfiguration('z_pos', default='0.1')

    # URDF via xacro
    robot_description = Command([
        'xacro ',
        PathJoinSubstitution([pkg_robot_description, 'urdf', 'robot.urdf.xacro'])
    ])

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments=[
            ('gz_args', [
                PathJoinSubstitution([pkg_robot_sim, 'worlds', world_file]),
                ' -r -v 4'
            ]),
        ]
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'diff_drive_robot',
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
        ],
        output='screen',
    )

    # Gazebo-ROS bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/model/diff_drive_robot/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # RViz2
    rviz = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([pkg_robot_description, 'rviz', 'robot.rviz'])],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # RL Node
    rl_node = Node(
        condition=IfCondition(use_rl),
        package='robot_controller',
        executable='rl_node',
        name='rl_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'model_path': model_path,
            'rl_config': rl_config,
            'camera_topic': camera_topic,
            'cmd_vel_topic': cmd_vel_topic,
            'inference_rate': 20.0,
            'action_type': 'continuous',
            'max_linear_speed': 2.0,
            'max_angular_speed': 4.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('world_file', default_value='obstacle_course.world', description='World file to load'),
        DeclareLaunchArgument('model_path', default_value='', description='Path to trained RL model'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('use_rl', default_value='true', description='Launch RL node'),
        DeclareLaunchArgument('rl_config', default_value='', description='Path to RL config file'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw', description='Camera image topic'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel', description='Command velocity topic'),
        DeclareLaunchArgument('x_pos', default_value='0.0'),
        DeclareLaunchArgument('y_pos', default_value='0.0'),
        DeclareLaunchArgument('z_pos', default_value='0.1'),

        robot_state_publisher,
        joint_state_publisher,
        gazebo,
        spawn_robot,
        bridge,
        rviz,
        rl_node,
    ])
