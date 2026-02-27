# STIV (Self-driving Thoroughly Intelligent Vehicle)

## Packages

- `stiv_control`: Control core functionality, written in Python.
- `stiv_perception`: Perception core functionality, written in Python.
- `stiv_navigation`: Navigation core functionality, written in C++. 
- `stiv_ros_interface_cpp`: ROS2 interface for C++ functionality.
- `stiv_ros_interface_py`: ROS2 interface for Python functionality.
- `stiv_msgs`: Language-agnostic ROS2 message and service definitions. 

## Installation

.bashrc aliases
```
alias make_venv='python3 -m venv --system-site-packages venv && touch venv/COLCON_IGNORE && source venv/bin/activate && rosdep install --from-paths src --ignore-src -r -y'
alias sd='source /opt/ros/jazzy/setup.bash && source venv/bin/activate && . install/setup.bash'
alias build='source /opt/ros/jazzy/setup.bash && source venv/bin/activate && python3 -m colcon build && . install/local_setup.bash'
```

Install the workspace and create a virtual environment.
```
mkdir -p stiv_ws/src
cd stiv_ws/src
git clone https://github.com/jftaggart02/stiv
git clone https://gitlab.com/utahstate/droge-robotics/general_research_code/rosmaster_r2_akm_driver  # for message definitions
cd ..
make_venv
build
```

Whenever you open a new terminal, run the following command (from the stiv_ws directory) to source the workspace and activate the virtual environment.
```
sd
```

## Network Setup

Activate lab computer hotspot and have jetson connect to it. Then make sure both have the same ROS_DOMAIN_ID.

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
