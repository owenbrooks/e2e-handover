// Adapted from https://github.com/yujinrobot/yujin_ocs/blob/devel/yocs_velocity_smoother/include/yocs_velocity_smoother/velocity_smoother_nodelet.hpp
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

#ifndef VELOCITY_SMOOTHER_HPP_
#define VELOCITY_SMOOTHER_HPP_


namespace velocity_smoother {

class VelocitySmoother
{
public:
  VelocitySmoother();

  ~VelocitySmoother()
  {
  }

  bool init(ros::NodeHandle& nh);
  void spin();

private:
  double speed_lim_v, accel_lim_v, decel_lim_v;
  double speed_lim_w, accel_lim_w, decel_lim_w;
  double decel_factor;

  double frequency;

  geometry_msgs::Twist last_cmd_vel;
  geometry_msgs::Twist  current_vel;
  geometry_msgs::Twist   target_vel;

  bool                 input_active;
  double                cb_avg_time;
  ros::Time            last_cb_time;
  std::vector<double> period_record; /**< Historic of latest periods between velocity commands */
  unsigned int             pr_next; /**< Next position to fill in the periods record buffer */

  ros::Subscriber raw_in_vel_sub;  /**< Incoming raw velocity commands */
  ros::Publisher  smooth_vel_pub;  /**< Outgoing smoothed velocity commands */

  void velocityCB(const geometry_msgs::Twist::ConstPtr& msg);

  double sign(double x)  { return x < 0.0 ? -1.0 : +1.0; };

  void apply_accel_limit(double accel_lim, double decel_lim, double period, 
  std::array<double, 3> target_values, std::array<double, 3> previous_values, 
  std::array<double*, 3> new_values);

  double median(std::vector<double> values) {
    // Return the median element of an doubles vector
    nth_element(values.begin(), values.begin() + values.size()/2, values.end());
    return values[values.size()/2];
  };

};

double bound_magnitude_within(double input, double limit);
bool twists_are_equal(geometry_msgs::Twist &left, geometry_msgs::Twist &right);
bool is_zero_velocity(geometry_msgs::Twist &msg);

} // namespace velocity_smoother

#endif /* VELOCITY_SMOOTHER_HPP_ */