# Deep Human-Robot Handover
## Force Thresholding Baseline Approach
`roslaunch robot_control force_baseline.launch`

## Data recording

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

Install additional dependencies by running `rosdep install --from-paths src --ignore-src -r -y` from your catkin workspace.

## Miscellaneous commands for reference
Viewing the robot urdf:
`roslaunch robot_control view_ur5_ft_grip_table.launch`

Gripper communication:
`rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0`

Force/torque sensor comms:
`rosrun robotiq_ft_sensor rq_sensor`

Camera:
`roslaunch realsense2_camera rs_camera.launch`

Testing recorder:
`rosrun robot_control record.py /camera/color/image_raw:=/image_publisher_1638368985896366349/image_raw`
`rosrun image_publisher image_publisher src/robot_control/test.png`