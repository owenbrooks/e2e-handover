#include <ros/ros.h>
#include <e2e_handover/velocity_smoother.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "velocity_smoother");
  ros::NodeHandle n("~");

  velocity_smoother::VelocitySmoother smoother;
  smoother.init(n);
  smoother.spin();

  return 0;
}