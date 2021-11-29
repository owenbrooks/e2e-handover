# How to run
- `roslaunch ur_gazebo ur5_bringup.launch`
- `roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true`
Optional:
- `roslaunch ur5_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur5_moveit_config)/launch/moveit.rviz`
- `rosrun robot_control test_move.py`

Viewing the robot urdf:
`roslaunch robot_control view_ur5_ft_grip_table.launch`