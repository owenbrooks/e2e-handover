#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <cmath>

double deg2rad(double deg) 
{
  return deg * M_PI / 180.0;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "position");
  ros::NodeHandle n("~");

  // ROS spinning must be running for the MoveGroupInterface to get information
  // about the robot's state. One way to do this is to start an AsyncSpinner
  // beforehand.
  ros::AsyncSpinner spinner(1);
  spinner.start();

  static const std::string PLANNING_GROUP = "manipulator";
  moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  double max_waiting_time = 20.0; // seconds
  const moveit::core::JointModelGroup* joint_model_group =
      move_group_interface.getCurrentState(max_waiting_time)->getJointModelGroup(PLANNING_GROUP);

  moveit::planning_interface::MoveGroupInterface::Plan my_plan;

  moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
  // base, shoulder, elbow, wrist1, wrist2, wrist3
  std::vector<double> joint_group_positions = {deg2rad(90.), deg2rad(-50.), deg2rad(100.), deg2rad(-230.), deg2rad(-90.), deg2rad(0.)};
  // Now, plan to the new joint space goal.
  move_group_interface.setJointValueTarget(joint_group_positions);

  bool success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  ROS_INFO_NAMED("Position", "Planning state: %s", success ? "" : "FAILED");
  
  char response;
  while (response != 'y') {
    std::cout << "Execute plan? (y/n): ";
    std::cin >> response;
  }
  ROS_INFO("Executing plan");
  move_group_interface.execute(my_plan);

  ros::shutdown();
  return 0;
}