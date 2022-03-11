#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from glob import glob
import os

def image_folder_publisher():
    image_pub = rospy.Publisher('/image', Image, queue_size=10)
    rospy.init_node('image_folder_publisher')
    rate = rospy.Rate(10) # hz
    folder_path = rospy.get_param('~image_folder')
    glob(folder_path)
    while not rospy.is_shutdown():
        rospy.loginfo(f"{folder_path}")
        # image_pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        image_folder_publisher()
    except rospy.ROSInterruptException:
        pass