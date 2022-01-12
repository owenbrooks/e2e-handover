# Deep Human-Robot Handover
## Force Thresholding Baseline Approach
Robot activates or releases the gripper when it detects a sufficient force in the z-axis.

`roslaunch robot_control force_baseline.launch`

Optional:
- `roslaunch ur5_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur5_moveit_config)/launch/moveit.rviz`

## Data recording
Data is stored in the `data` directory.

Pressing 'r' begins or ends a recording session, identified by a timestamp. Images and a csv file for each session are stored in a folder. Pressing 'shift' toggles the gripper state manually.

`roslaunch robot_control inference.launch`

Optionally, record the raw ROS messages with `rosbag record rosout /camera/color/image_raw robotiq_ft_wrench Robotiq2FGripperRobotInput /Robotiq2FGripperRobotOutput`

## Training
As the docker container doesn't have CUDA support, this should be done outside the container.

`pip install wandb`

`pip install torch`

`python3 src/robot_control/train.py --session 2021-12-01-15:30:36`

## Inference

`roslaunch robot_control inference.launch`

# Hardware Setup
- UR5 Robot
- Intel Realsense 2 RGBD camera
- Robotiq 2F85 gripper
- Robotiq FT300 force-torque sensor

# Dependencies
## Using Docker

- Install docker (using `sudo apt install docker.io`)
- (optional) Install [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
- Run `./run_docker.sh`

## Building from source
- [ROS Noetic](http://wiki.ros.org/noetic/Installation)
- [Universal Robots ROS Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) (Follow *Building* instructions)
- [fmauch Robot Descriptions](https://github.com/fmauch/universal_robot) (Should be installed as per Universal Robots instructions)
- [Robotiq drivers (updated to work with ROS Noetic)](https://github.com/jr-robotics/robotiq.git) (Clone into `src` directory)
- [Realsense Camera ROS Drivers](https://github.com/IntelRealSense/realsense-ros)

Aside from ROS itself, these dependencies can be downloaded automatically by `vcstool` (install with `sudo apt install python3-vcstool`), then go to `src` directory and run `vcs import < e2e.rosinstall`.

To update the dependency list when adding more packages in the future, use `vcs export > e2e.rosinstall`.

Ensure pip is installed via `sudo apt-get install python3-pip`.

Install additional dependencies by running `rosdep install --from-paths src --ignore-src -r -y` from your catkin workspace.

# Miscellaneous commands for reference
Bring up communication with the robot:
`roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.0.0.2`

Viewing the robot urdf:
`roslaunch robot_control view_ur5_ft_grip_table.launch`

Gripper communication:
`rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0`

Force/torque sensor comms:
`rosrun robotiq_ft_sensor rq_sensor`

Camera:
`roslaunch realsense2_camera rs_camera.launch`
or `roslaunch camera_driver realsense_driver.launch`

Testing recorder:
- Place an image in data/test.png
- `roslaunch robot_control test_recording.launch`

Testing inference:
- `roslaunch robot_control test_inference.launch`

Running in gazebo
- `roslaunch ur_gazebo ur5_bringup.launch` / `roslaunch robot_control ur5_bringup_gazebo.launch`
- `roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true`

Combine datasets:
`python3 src/robot_control/data_prep.py --session 2021-12-17-tactile combine -l 2021-12-17-00:55:09 2021-12-17-01:01:44`

## Pairing controller
- For DualShock PS4 controller, press and hold both the Share and PS buttons.
- Open Bluetooth settings and click on 'Wireless controller' to pair.
