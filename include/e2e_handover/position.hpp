// Adapted from https://github.com/yujinrobot/yujin_ocs/blob/devel/yocs_position/include/yocs_position/position_nodelet.hpp
#include <ros/ros.h>

#ifndef POSITION_HPP_
#define POSITION_HPP_


namespace position {

class Position
{
public:
  Position();

  ~Position()
  {
  }

  bool init(ros::NodeHandle& nh);
  void spin();

private:
  ros::ServiceServer move_service;
  ros::ServiceServer plan_service;

  std::array<double, 6> target_joint_values_;

};

} // namespace position

#endif /* POSITION_HPP_ */