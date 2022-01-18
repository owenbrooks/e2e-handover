#include <ros/ros.h>
#include <robot_control/velocity_smoother.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "velocity_smoother");

  ros::NodeHandle n("~");

  ros::Rate loop_rate(10);

  velocity_smoother::VelocitySmoother smoother;
  smoother.init(n);
  smoother.spin();

  return 0;
}