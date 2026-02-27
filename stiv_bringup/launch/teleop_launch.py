from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch joy_node and teleop_node for teleoperation."""

    # Joy node for joystick input
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="joy_node",
        output="screen",
    )

    # Teleop node to convert joy messages to AkmControl
    teleop_node = Node(
        package="stiv_ros_interface_py",
        executable="teleop_node",
        name="teleop_node",
        output="screen",
        parameters=[
            {"joy_topic": "/joy"},
            {"control_topic": "/rm1/movement_control"},
            {"velocity_axis": 1},
            {"steering_axis": 2},
            {"max_velocity": 0.7},
            {"max_steering_angle": 45},
        ],
    )

    return LaunchDescription(
        [
            joy_node,
            teleop_node,
        ]
    )
