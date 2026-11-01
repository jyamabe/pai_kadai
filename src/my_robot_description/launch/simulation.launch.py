import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory(
        'my_robot_description'
    )

    urdf_file = os.path.join(
        package_dir,
        'urdf',
        'color_sorting_robot.urdf'
    )


    world_file = os.path.join(
        package_dir,
        'worlds',
        'sorting_world.world'
    )


    with open(urdf_file, 'r') as file:
        robot_description = file.read()

    gazebo_launch = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gazebo_launch),
            launch_arguments={
                'world': world_file
            }.items()
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'robot_description': robot_description,
                    'use_sim_time': True
                }
            ]
        ),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity',
                'color_sorting_robot',
                '-topic',
                'robot_description',
                '-x', '-1.0',
                '-y', '0.0',
                '-z', '0.0',
                '-Y', '0.0'
            ],
            output='screen'
        )
    ])
