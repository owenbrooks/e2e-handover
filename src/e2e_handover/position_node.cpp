#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

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
  const moveit::core::JointModelGroup* joint_model_group =
      move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

  // Visualization
  // ^^^^^^^^^^^^^
  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools("base");
  visual_tools.deleteAllMarkers();

  // Remote control is an introspection tool that allows users to step through a high level script
  // via buttons and keyboard shortcuts in RViz
  visual_tools.loadRemoteControl();

  // RViz provides many types of markers, in this demo we will use text, cylinders, and spheres
  Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
  text_pose.translation().z() = 1.0;
  visual_tools.publishText(text_pose, "MoveGroupInterface Demo", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();

  while (true) {
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
    std::vector<double> joint_group_positions;
    current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);

    // Now, let's modify one of the joints, plan to the new joint space goal and visualize the plan.
    joint_group_positions[0] = -2*M_PI / 6;  // -1/6 turn in radians
    move_group_interface.setJointValueTarget(joint_group_positions);

    bool success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Visualizing plan (joint space goal) %s", success ? "" : "FAILED");

    auto remote_control = visual_tools.getRemoteControl();
    ROS_INFO("Press 'continue' to re-plan, or press 'next' to execute. (in the RvizVisualToolsGui window)");
    remote_control->waitForNextStep();

    if (!remote_control->getAutonomous()) {
        ROS_INFO("Executing plan");
        move_group_interface.execute(my_plan);
    } else {
      remote_control->setAutonomous(false);
    }

    // // Visualize the plan in RViz
    // visual_tools.deleteAllMarkers();
    // visual_tools.publishText(text_pose, "Joint Space Goal", rvt::WHITE, rvt::XLARGE);
    // visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
    // visual_tools.trigger();

  }

  ros::shutdown();
  return 0;
}