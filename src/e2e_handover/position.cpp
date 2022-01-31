#include <ros/ros.h>
#include <e2e_handover/position.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

namespace position {

Position::Position()
{
};

void Position::spin()
{
  ros::Rate spin_rate(frequency);

  namespace rvt = rviz_visual_tools;
  moveit_visual_tools::MoveItVisualTools visual_tools("manipulator");
  visual_tools.deleteAllMarkers();
  visual_tools.loadRemoteControl();
  visual_tools.publishText(text_pose, "MoveGroupInterface Demo", rvt::WHITE, rvt::XLARGE);
  visual_tools.trigger();
  visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to start the demo");

  target_joint_values_ = {-0.037211508523346204, -0.14613448707173848, 0.00042289139595208525, 
                          0.0618242143312564, 4.002682490433784e-05, 0.1302630262299358} // joint angles in radians

  

  while (ros::ok())
  {   
    ROS_WARN_STREAM("Position node");

    ros::spinOnce();
    spin_rate.sleep();
  }
}

bool Position::init(ros::NodeHandle& nh)
{
  // Services
  move_service = n.advertiseService("add_two_ints", add);
  plan_service = n.advertiseService("add_two_ints", add);

  static const std::string PLANNING_GROUP = "manipulator";
  moveit::planning_interface::MoveGroupInterface move_group_interface(PLANNING_GROUP);

  return true;
}

} // namespace position
