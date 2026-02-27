# STIV (Self-driving Thoroughly Intelligent Vehicle)

## Packages

- `stiv_control`: Control core functionality, written in Python.
- `stiv_perception`: Perception core functionality, written in Python.
- `stiv_navigation`: Navigation core functionality, written in C++. 
- `stiv_ros_interface_cpp`: ROS2 interface for C++ functionality.
- `stiv_ros_interface_py`: ROS2 interface for Python functionality.
- `stiv_msgs`: Language-agnostic ROS2 message and service definitions. 

## Joystick Setup

Install joystick package.
```
sudo apt install ros-jazzy-joy
```

Plug in your joystick and run the following command to check if it is recognized by the system. You should see `js0` listed.
```
ls /dev/input
```

Test joystick
```
sudo apt-get install joystick
sudo jstest /dev/input/js0
```

Test joystick with ROS
```
ros2 run joy joy_node
ros2 topic echo /joy  # in another terminal
```

## Run Teleop Node

Make sure you have the `stiv_ros_interface_py` package built and sourced. Then run the launch file:
```
ros2 launch stiv_bringup teleop_launch.py
```
