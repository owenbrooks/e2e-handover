#!/usr/bin/env python3
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import Twist

class JoystickTwist:
    def __init__(self):
        rospy.init_node('joystick_twist', anonymous=False)
        rospy.Subscriber("joy", sensor_msgs.msg.Joy, self.joystick_callback)
        self.twist_pub = rospy.Publisher("/twist_cmd_raw", Twist, queue_size=10)
        self.twist_msg = Twist()

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

        twist_msg = Twist()

        # Populate twist msg with joystick values ranging from -1.0 to 1.0
        twist_msg.linear.x = -deadband(left_x)
        twist_msg.linear.y = deadband((trig_r-1.0)/2.0 - (trig_l-1.0)/2.0)
        twist_msg.linear.z = deadband(left_y)
        twist_msg.angular.x =  deadband(right_y)
        twist_msg.angular.y = -deadband(right_x)
        twist_msg.angular.z = bump_r - bump_l

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