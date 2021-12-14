#!/usr/bin/env python3
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import TwistStamped

class JoystickTwist:
    def __init__(self):
        rospy.init_node('joystick_twist', anonymous=False)
        rospy.Subscriber("joy", sensor_msgs.msg.Joy, self.joystick_callback)
        self.twist_pub = rospy.Publisher("twist_cmd", TwistStamped, queue_size=10)
        self.twist_msg = TwistStamped

    def spin(self):
        # Publish the latest control message at 30Hz
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.twist_pub.publish(self.twist_msg)
            rate.sleep()

    def joystick_callback(self,data):
        # Get all the axes and button values
        left_x, left_y, trig_l, right_x, right_y, trig_r, dpad_x, dpad_y = data.axes
        btn_a, btn_b, btn_x, btn_y, bump_l, bump_r, back, menu, _, stick_l, stick_r, _, _ = data.buttons

        twist_msg = TwistStamped()
        twist_msg.header.frame_id = 'camera_color_optical_frame'

        # Translation
        twist_msg.twist.linear.x = -deadband(left_x)
        twist_msg.twist.linear.y = - deadband(left_y)
        twist_msg.twist.linear.z = deadband(trig_l/2.0 - trig_r/2.0)
        # Rotation
        twist_msg.twist.angular.x =  deadband(right_y)
        twist_msg.twist.angular.y = - deadband(right_x)
        twist_msg.twist.angular.z = bump_r - bump_l

        self.twist_msg = twist_msg

def deadband(var,band=0.2):
    var = max(-1.0,min(var,1.0))

    if var > band:
        return (var-band) / (1.0-band)

    if var < -band:
        return (var+band) / (1.0-band)
    return 0.0

if __name__ == "__main__":
    node = JoystickTwist()
    node.spin()