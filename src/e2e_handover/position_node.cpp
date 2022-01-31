#include <ros/ros.h>
#include <e2e_handover/position.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "position");
  ros::NodeHandle n("~");

  // ROS spinning must be running for the MoveGroupInterface to get information
  // about the robot's state. One way to do this is to start an AsyncSpinner
  // beforehand.
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // Setup
  // ^^^^^
  //
  static const std::string PLANNING_GROUP = "manipulator";
  moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  const moveit::core::JointModelGroup* joint_model_group =
      move_group_interface.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

  // Visualization
  // ^^^^^^^^^^^^^
  //
  // The package MoveItVisualTools provides many capabilities for visualizing objects, robots,
  // and trajectories in RViz as well as debugging tools such as step-by-step introspection of a script.
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

  // Batch publishing is used to reduce the number of messages being sent to RViz for large visualizations
  visual_tools.trigger();

  // Getting Basic Information
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // We can print the name of the reference frame for this robot.
  ROS_INFO_NAMED("tutorial", "Planning frame: %s", move_group_interface.getPlanningFrame().c_str());

  // We can also print the name of the end-effector link for this group.
  ROS_INFO_NAMED("tutorial", "End effector link: %s", move_group_interface.getEndEffectorLink().c_str());

  // We can get a list of all the groups in the robot:
  ROS_INFO_NAMED("tutorial", "Available Planning Groups:");
  std::copy(move_group_interface.getJointModelGroupNames().begin(),
            move_group_interface.getJointModelGroupNames().end(), std::ostream_iterator<std::string>(std::cout, ", "));

  // Start the demo
  // ^^^^^^^^^^^^^^^^^^^^^^^^^
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  // .. _move_group_interface-planning-to-pose-goal:
  //
  // Planning to a Pose goal
  // ^^^^^^^^^^^^^^^^^^^^^^^
  // We can plan a motion for this group to a desired pose for the
  // end-effector.
  geometry_msgs::Pose target_pose1;
  target_pose1.orientation.w = 1.0;
  target_pose1.position.x = 0.28;
  target_pose1.position.y = -0.2;
  target_pose1.position.z = 0.5;
  move_group_interface.setPoseTarget(target_pose1);

  // Now, we call the planner to compute the plan and visualize it.
  // Note that we are just planning, not asking move_group_interface
  // to actually move the robot.
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;

  // Planning to a joint-space goal
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // Let's set a joint space goal and move towards it.  This will replace the
  // pose target we set above.
  //
  // To start, we'll create an pointer that references the current robot's state.
  // RobotState is the object that contains all the current position/velocity/acceleration data.
  moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState();
  //
  // Next get the current set of joint values for the group.
  std::vector<double> joint_group_positions;
  current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);

  // Now, let's modify one of the joints, plan to the new joint space goal and visualize the plan.
  joint_group_positions[0] = -tau / 6;  // -1/6 turn in radians
  move_group_interface.setJointValueTarget(joint_group_positions);

  // We lower the allowed maximum velocity and acceleration to 5% of their maximum.
  // The default values are 10% (0.1).
  // Set your preferred defaults in the joint_limits.yaml file of your robot's moveit_config
  // or set explicit factors in your code if you need your robot to move faster.
  move_group_interface.setMaxVelocityScalingFactor(0.05);
  move_group_interface.setMaxAccelerationScalingFactor(0.05);

  bool success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  ROS_INFO_NAMED("tutorial", "Visualizing plan 2 (joint space goal) %s", success ? "" : "FAILED");


  // Visualize the plan in RViz
  visual_tools.deleteAllMarkers();
  visual_tools.publishText(text_pose, "Joint Space Goal", rvt::WHITE, rvt::XLARGE);
  visual_tools.publishTrajectoryLine(my_plan.trajectory_, joint_model_group);
  visual_tools.trigger();
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // Finally, to execute the trajectory stored in my_plan, you could use the following method call:
  // Note that this can lead to problems if the robot moved in the meanwhile.
  move_group_interface.execute(my_plan);

  ros::shutdown();
  return 0;
}