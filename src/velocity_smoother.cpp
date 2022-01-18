// Adapted from https://github.com/yujinrobot/yujin_ocs/blob/devel/yocs_velocity_smoother/src/velocity_smoother_nodelet.cpp

#include <ros/ros.h>
#include <robot_control/velocity_smoother.hpp>

#define PERIOD_RECORD_SIZE    5

namespace velocity_smoother {

VelocitySmoother::VelocitySmoother()
: input_active(false)
, pr_next(0)
{
};

void VelocitySmoother::velocityCB(const geometry_msgs::Twist::ConstPtr& msg)
{
  // Estimate command frequency; we do this continuously as it can be very different depending on the
  // publisher type, and we don't want to impose extra constraints to keep this package flexible
  if (period_record.size() < PERIOD_RECORD_SIZE)
  {
    period_record.push_back((ros::Time::now() - last_cb_time).toSec());
  }
  else
  {
    period_record[pr_next] = (ros::Time::now() - last_cb_time).toSec();
  }

  pr_next++;
  pr_next %= period_record.size();
  last_cb_time = ros::Time::now();

  if (period_record.size() <= PERIOD_RECORD_SIZE/2)
  {
    // wait until we have some values; make a reasonable assumption (10 Hz) meanwhile
    cb_avg_time = 0.1;
  }
  else
  {
    // enough; recalculate with the latest input
    cb_avg_time = median(period_record);
  }

  input_active = true;

  // Bound speed with the maximum values
  target_vel.linear.x  = bound_magnitude_within(msg->linear.x, speed_lim_v);
  target_vel.linear.y  = bound_magnitude_within(msg->linear.y, speed_lim_v);
  target_vel.linear.z  = bound_magnitude_within(msg->linear.z, speed_lim_v);
  target_vel.angular.x = bound_magnitude_within( msg->angular.x, speed_lim_w);
  target_vel.angular.y = bound_magnitude_within( msg->angular.y, speed_lim_w);
  target_vel.angular.z = bound_magnitude_within( msg->angular.z, speed_lim_w);
}

void VelocitySmoother::apply_accel_limit(double accel_lim, double decel_lim, 
  double period, std::array<double, 3> target_values, 
  std::array<double, 3> previous_values, std::array<double*, 3> new_values)
{
  for (int i = 0; i < 3; i++) {
    double target = target_values[i];
    double previous = previous_values[i];
    double *cmd = new_values[i];

    double v_inc = target - previous;
  
    double max_v_inc = ((v_inc*target > 0.0)?accel_lim:decel_lim)*period;

    if (std::abs(v_inc) > max_v_inc)
    {
      // we must limit linear velocity
      *cmd  = previous + sign(v_inc)*max_v_inc;
    }
  }
}

void VelocitySmoother::spin()
{
  double period = 1.0/frequency;
  ros::Rate spin_rate(frequency);

  while (ros::ok())
  {   
    if ((input_active == true) && (cb_avg_time > 0.0) &&
        ((ros::Time::now() - last_cb_time).toSec() > std::min(3.0*cb_avg_time, 0.5)))
    {
      // Velocity input not active anymore; normally last command is a zero-velocity one, but reassure
      // this, just in case something went wrong with our input, or we just forgot good manners...
      // Issue #2, extra check in case cb_avg_time is very big, for example with several atomic commands
      // The cb_avg_time > 0 check is required to deal with low-rate simulated time, that can make that
      // several messages arrive with the same time and so lead to a zero median
      input_active = false;
      if (!is_zero_velocity(target_vel))
      {
        ROS_WARN_STREAM("Velocity Smoother: input became inactive leaving us a non-zero target velocity ("
              << target_vel.linear.x << ", " << target_vel.angular.z << "), zeroing...");
        target_vel = geometry_msgs::Twist(); // sends 0 velocity
      }
    }

    geometry_msgs::TwistPtr cmd_vel;

    if (!twists_are_equal(target_vel, last_cmd_vel))
    {
      // Try to reach target velocity ensuring that we don't exceed the acceleration limits
      cmd_vel.reset(new geometry_msgs::Twist(target_vel));

      double v_inc, w_inc, max_v_inc, max_w_inc;

      std::array<double, 3> linear_targets = {target_vel.linear.x, target_vel.linear.y, target_vel.linear.z};
      std::array<double, 3> angular_targets = {target_vel.angular.x, target_vel.angular.y, target_vel.angular.z};
      std::array<double, 3> linear_last_cmds = {last_cmd_vel.linear.x, last_cmd_vel.linear.y, last_cmd_vel.linear.z};
      std::array<double, 3> angular_last_cmds = {last_cmd_vel.angular.x, last_cmd_vel.angular.y, last_cmd_vel.angular.z};
      std::array<double*, 3> linear_cmds = {&(cmd_vel->linear.x), &(cmd_vel->linear.y), &(cmd_vel->linear.z)};
      std::array<double*, 3> angular_cmds = {&(cmd_vel->angular.x), &(cmd_vel->angular.y), &(cmd_vel->angular.z)};

      // Modify the cmd message to send velocities that are within accel_lim from previous command
      apply_accel_limit(accel_lim_v, decel_lim_v, period, linear_targets, linear_last_cmds, linear_cmds);
      apply_accel_limit(accel_lim_w, decel_lim_w, period, angular_targets, angular_last_cmds, angular_cmds);

      smooth_vel_pub.publish(cmd_vel);
      last_cmd_vel = *cmd_vel;
    }
    else if (input_active == true)
    {
      // We already reached target velocity; just keep resending last command while input is active
      cmd_vel.reset(new geometry_msgs::Twist(last_cmd_vel));
      smooth_vel_pub.publish(cmd_vel);
    }

    ros::spinOnce();
    spin_rate.sleep();
  }
}

bool VelocitySmoother::init(ros::NodeHandle& nh)
{
  // Optional parameters
  nh.param("frequency",      frequency,     20.0);
  nh.param("decel_factor",   decel_factor,   1.0);

  // Mandatory parameters
  if ((nh.getParam("speed_lim_v", speed_lim_v) == false) ||
      (nh.getParam("speed_lim_w", speed_lim_w) == false))
  {
    ROS_ERROR("Missing velocity limit parameter(s)");
    return false;
  }

  if ((nh.getParam("accel_lim_v", accel_lim_v) == false) ||
      (nh.getParam("accel_lim_w", accel_lim_w) == false))
  {
    ROS_ERROR("Missing acceleration limit parameter(s)");
    return false;
  }

  // Deceleration can be more aggressive, if necessary
  decel_lim_v = decel_factor*accel_lim_v;
  decel_lim_w = decel_factor*accel_lim_w;

  // Publishers and subscribers
  raw_in_vel_sub  = nh.subscribe("raw_cmd_vel", 10, &VelocitySmoother::velocityCB, this);
  smooth_vel_pub  = nh.advertise<geometry_msgs::Twist>("smooth_cmd_vel", 1);

  return true;
}

double bound_magnitude_within(double input, double limit) 
{
  double output  = input > 0.0 ? std::min(input, limit) : std::max(input, -limit);
  return output;
}

bool twists_are_equal(geometry_msgs::Twist &left, geometry_msgs::Twist &right)
{
  return left.linear.x == right.linear.x && left.linear.y == right.linear.y && 
  left.linear.z == right.linear.z && left.angular.x == right.angular.x &&
  left.angular.y == right.angular.y && left.angular.z == right.angular.z;
}

bool is_zero_velocity(geometry_msgs::Twist &msg)
{
  return msg.linear.x == 0.0 && msg.linear.y == 0.0 && msg.linear.z == 0.0 
    && msg.angular.x == 0.0 && msg.angular.y == 0.0 && msg.angular.z == 0.0;
}

} // namespace velocity_smoother
