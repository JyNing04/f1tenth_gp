from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='f1tenth_ros2',
            executable='joystick_ros2',
            output='screen'
        ),
        Node(
            package='f1tenth_ros2',
            executable='teleop_joy',
            output='screen'
        ),
    ])