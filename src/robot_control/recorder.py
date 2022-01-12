#!/usr/bin/env python
import os
import csv
import rospy
from geometry_msgs.msg import Twist
from datetime import datetime
import cv2
import numpy as np
from robot_control import tactile

# Class to record data 
# Data is stored in 'data/${SESSION_ID}' folder, where SESSION_ID is unique timestamp
# Images are stored in that folder with index and session id as name
# Each session folder has a csv with numerical data and gripper state associated with image name

class Recorder():
    def __init__(self, record_tactile):
        # self.twist_sub = rospy.Subscriber('/twist_controller/command', Twist, self.force_callback)
        self.twist_sub = rospy.Subscriber('/twist_cmd_raw', Twist, self.twist_callback)

        self.is_recording = False
        self.session_id = ""
        self.session_image_count = 0

        self.twist_array = 6*[0.0]

        self.record_tactile = record_tactile

        # Create folder for storing recorded images and the csv with numerical/annotation data
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../../data')
        data_dir_exists = os.path.exists(data_dir)
        if not data_dir_exists:
            os.makedirs(data_dir)

    def twist_callback(self, twist_msg):
        self.twist_array = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]

    def record_row(self, image, wrench, gripper_is_open, tactile_readings=[]):
        if self.is_recording:
            # Save image as png
            current_dirname = os.path.dirname(__file__)
            image_name = str(self.session_image_count) + '_' + self.session_id + '.png'
            self.session_image_count += 1
            image_path = os.path.join(current_dirname, '../../data', self.session_id, 'images', image_name)
            cv2.imwrite(image_path, image)

            # Append numerical data and annotation to the session csv
            csv_path = os.path.join(current_dirname, '../../data', self.session_id, self.session_id + '.csv')
            with open(csv_path, 'a+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ')
                row = [image_name, gripper_is_open] + wrench.tolist() + self.twist_array

                if self.record_tactile:
                    row += tactile_readings

                datawriter.writerow(row)

    def start_recording(self):
        if self.is_recording:
            rospy.loginfo("Already recording. Session: " + self.session_id)
        else:
            self.is_recording = True
            self.session_image_count = 0
            # Create folder for storing recorded images and the session csv
            self.session_id = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            current_dirname = os.path.dirname(__file__)
            session_dir = os.path.join(current_dirname, '../../data', self.session_id)
            image_dir = os.path.join(session_dir, 'images')
            os.makedirs(image_dir)
            # Create csv file for recording numerical data and annotation in the current session
            fname = os.path.join(session_dir, self.session_id + '.csv')
            with open(fname, 'w+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                
                wrench_header = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
                twist_header = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
                header = ['image_id', 'gripper_is_open'] + wrench_header + twist_header
                if self.record_tactile:
                    header += tactile.papillarray_keys
            
                datawriter.writerow(header)

            rospy.loginfo("Started recording. Session: " + self.session_id)

    def stop_recording(self):
        if not self.is_recording:
            rospy.loginfo("Hadn't yet started recording.")
        else:
            self.is_recording = False
            rospy.loginfo("Finished recording. Session: " + self.session_id)
            rospy.loginfo("Recorded " + str(self.session_image_count) + " images")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_gripper(self):
        self.toggle_key_pressed = True
    
if __name__ == "__main__":
    try: 
        recorder = Recorder(True)
        recorder.start_recording()
        recorder.record_row(np.array([1]), np.zeros(6), False, np.zeros(10*6).tolist())
    except KeyboardInterrupt:
        pass