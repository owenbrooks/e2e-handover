# Deep Human-Robot Handover
## Force Thresholding Baseline Approach
Robot activates or releases the gripper when it detects a sufficient force in the z-axis.

`roslaunch robot_control force_baseline.launch`

## Data recording
Data is stored in the `data` directory.

Pressing 'r' begins or ends a recording session, identified by a timestamp. Images and a csv file for each session are stored in a folder. Pressing 'shift' toggles the gripper state manually.

`roslaunch robot_control recording.launch`

## Training
`python src/train.py /home/owen/srp_ws/src/e2e-handover/data/2021-12-01-15:30:36`

## Running in gazebo
- `roslaunch ur_gazebo ur5_bringup.launch` / `roslaunch robot_control ur5_bringup_gazebo.launch`
- `roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true`

Optional:
- `roslaunch ur5_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur5_moveit_config)/launch/moveit.rviz`

# Hardware Setup
- UR5 Robot
- Intel Realsense 2 RGBD camera
- Robotiq 2F85 gripper
- Robotiq FT300 force-torque sensor

# Dependencies
- [ROS Melodic](http://wiki.ros.org/melodic/Installation)
- [Universal Robots ROS Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) (Follow *Building* instructions)
- [fmauch Robot Descriptions](https://github.com/fmauch/universal_robot) (Should be installed as per Universal Robots instructions)
- [Robotiq drivers](https://github.com/ros-industrial/robotiq) (Clone into `src` directory)
- [Realsense Camera ROS Drivers](https://github.com/IntelRealSense/realsense-ros)

Aside from ROS itself, these dependencies can be downloaded automatically by `vcstool` (install with `sudo apt install python3-vcstool`), then go to `src` directory and run `vcs import < e2e.rosinstall`
To update the dependency list when adding more packages in the future, use `vcs export > e2e.rosinstall`

Ensure pip is installed via `sudo apt-get install python-pip`.

Install additional dependencies by running `rosdep install --from-paths src --ignore-src -r -y` from your catkin workspace.

## Miscellaneous commands for reference
Bring up communication with the robot:
`roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=IP_OF_THE_ROBOT`

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
- `rosrun robot_control record.py /camera/color/image_raw:=/image_publisher_1638368985896366349/image_raw`

- `rosrun image_publisher image_publisher src/robot_control/test.png`

## Docker:

Build the container:
`docker build . -t e2e`

Initialise with nvidia GPU support (requires installing `nvidia-docker2` as specified in [nvidia docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)):

`docker run -e DISPLAY -e TERM
    --privileged 
    -v "/dev:/dev:rw"
    -v "${HOME}:${HOME}:rw"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    --runtime=nvidia
    --net=host
    --hostname e2e
    --name e2e
    --entrypoint /ros_entrypoint.sh
    -d e2e /usr/bin/tail -f /dev/null `

Initialise without GPU support:

`docker run -e DISPLAY -e TERM
    --privileged 
    -v "/dev:/dev:rw"
    -v "${HOME}:${HOME}:rw"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    --net=host
    --hostname e2e
    --name e2e
    --entrypoint /ros_entrypoint.sh
    -d e2e /usr/bin/tail -f /dev/null `

Drop into terminal:
`docker exec -it e2e bash`

Kill container:
`docker kill e2e`

Remove container:
`docker rm e2e`

Kill and remove container:
`docker kill e2e && docker rm e2e`

Start stopped container:
`docker start e2e`