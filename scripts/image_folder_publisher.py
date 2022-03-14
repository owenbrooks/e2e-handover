#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import glob
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError


def image_folder_publisher():
    cv_bridge = CvBridge() # for converting ROS image messages to OpenCV images or vice versa

    rospy.init_node('image_folder_publisher')
    image_topic = rospy.get_param('~image_topic')
    image_pub = rospy.Publisher(image_topic, Image, queue_size=10)
    folder_path = rospy.get_param('~image_folder')
    spin_rate = rospy.get_param('~rate')
    rate = rospy.Rate(spin_rate) # hz
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*'))) # alphabetical sorting
    index = 0
    print(f"Publishing {len(image_paths)} images from folder {folder_path}")
    while not rospy.is_shutdown() and index != len(image_paths):
        img_path = image_paths[index]
        rospy.loginfo(f"{img_path}")
        img = cv2.imread(img_path)
        img_msg = cv_bridge.cv2_to_imgmsg(img, encoding='passthrough')
        # print(img)
        image_pub.publish(img_msg)
        index += 1
        rate.sleep()

    print(f"Finished publishing all images in {folder_path}")

if __name__ == '__main__':
    try:
        image_folder_publisher()
    except rospy.ROSInterruptException:
        pass