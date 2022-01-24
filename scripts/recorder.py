#!/usr/bin/env python3
import csv
import cv2
from datetime import datetime
from e2e_handover.sensor_manager import SensorManager
from e2e_handover import tactile
from geometry_msgs.msg import Twist
from pynput import keyboard
import os
from robotiq_2f_gripper_control.msg import  _Robotiq2FGripper_robot_input as inputMsg
import rospkg
import rospy

# Class to record data 
# Data is stored in 'data/${SESSION_ID}' folder, where SESSION_ID is unique timestamp
# Images are stored in that folder with index and session id as name
# Each session folder has a csv with numerical data and gripper state associated with image name

class Recorder():
    def __init__(self):
        rospy.init_node("recorder")
        self.twist_sub = rospy.Subscriber('/twist_cmd_raw', Twist, self.twist_callback)
        self.filtered_twist_sub = rospy.Subscriber('/twist_cmd_filtered', Twist, self.filtered_twist_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)

        self.is_recording = False
        self.session_id = ""
        self.row_count = 0

        self.twist_array = 6*[0.0]
        self.filtered_twist_array = 6*[0.0]

        self.gripper_is_open = False

        self.recording_params = rospy.get_param('~recording')
        self.sensor_manager = SensorManager(self.recording_params)

        # Create folder for storing recorded images and the csv with numerical/annotation data
        rospack = rospkg.RosPack()
        package_dir = rospack.get_path('e2e_handover')
        self.data_dir = os.path.join(package_dir, 'data')
        data_dir_exists = os.path.exists(self.data_dir)
        if not data_dir_exists:
            os.makedirs(self.data_dir)

    def gripper_state_callback(self, gripper_input_msg):
        position_requested = gripper_input_msg.gPR
        # should have value 0x00 for open, 0xFF for closed
        if position_requested == 0x00:
            self.gripper_is_open = True

    def twist_callback(self, twist_msg):
        self.twist_array = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]
    
    def filtered_twist_callback(self, twist_msg):
        self.filtered_twist_array = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]

    def save_image(self, image, count, identifier):
        # Save image as png
        image_name = f"{count}_{self.session_id}.png"
        rel_path = os.path.join(identifier, image_name)
        image_path = os.path.join(self.data_dir, self.session_id, rel_path)
        cv2.imwrite(image_path, image)

        return rel_path

    def record_row(self):
        if self.recording_params['use_rgb_1']:       
            rel_path_rgb_1 = self.save_image(self.sensor_manager.img_rgb_1, self.row_count, 'image_rgb_1')
        if self.recording_params['use_rgb_2']:
            rel_path_rgb_2 = self.save_image(self.sensor_manager.img_rgb_2, self.row_count, 'image_rgb_2')
        
        # Append numerical data and annotation to the session csv
        csv_path = os.path.join(self.data_dir, self.session_id, self.session_id + '.csv')
        with open(csv_path, 'a+') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=' ')
            row = [self.gripper_is_open] + self.twist_array + self.filtered_twist_array

            if self.recording_params['use_rgb_1']:
                row.append(rel_path_rgb_1)
            if self.recording_params['use_rgb_2']:
                row.append(rel_path_rgb_2)
            if self.recording_params['use_force']:
                row += self.sensor_manager.raw_wrench_reading.tolist()

            if self.recording_params['use_tactile']:
                row += self.sensor_manager.tactile_1_readings
                row += self.sensor_manager.tactile_2_readings

            datawriter.writerow(row)
            self.row_count += 1

    def start_recording(self):
        if self.is_recording:
            rospy.loginfo("Already recording. Session: " + self.session_id)
        else:
            self.sensor_manager.activate()
            self.is_recording = True
            self.row_count = 0
            # Create folder for storing recorded images and the session csv
            self.session_id = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            session_dir = os.path.join(self.data_dir, self.session_id)

            if self.recording_params['use_rgb_1']:
                image_rgb_1_dir = os.path.join(session_dir, 'image_rgb_1')
                os.makedirs(image_rgb_1_dir)
            if self.recording_params['use_rgb_2']:
                image_rgb_2_dir = os.path.join(session_dir, 'image_rgb_2')
                os.makedirs(image_rgb_2_dir)

            # Create csv file for recording numerical data and annotation in the current session
            fname = os.path.join(session_dir, self.session_id + '.csv')
            with open(fname, 'w+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                
                wrench_header = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
                twist_header = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
                filtered_twist_header = ['vx_filt', 'vy_filt', 'vz_filt', 'wx_filt', 'wy_filt', 'wz_filt']
                
                header = ['gripper_is_open'] + twist_header + filtered_twist_header
                
                if self.recording_params['use_rgb_1']:
                    header.append('image_rgb_1')
                if self.recording_params['use_rgb_2']:
                    header.append('image_rgb_2')
                if self.recording_params['use_force']:
                    header += wrench_header

                if self.recording_params['use_tactile']:
                    header += tactile.papillarray_keys
                    header += tactile.papillarray_keys
            
                datawriter.writerow(header)

            rospy.loginfo("Started recording. Session: " + self.session_id)

    def stop_recording(self):
        if not self.is_recording:
            rospy.loginfo("Hadn't yet started recording.")
        else:
            self.is_recording = False
            self.sensor_manager.deactivate()
            rospy.loginfo(f"Finished recording. Session: {self.session_id}")
            rospy.loginfo(f"Recorded {self.row_count + 1} frames")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'r': # r key to start/stop recording
                self.toggle_recording()
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            pass

    def run(self):
        rate = rospy.Rate(30)

        # keyboard input
        key_listener = keyboard.Listener(on_press=self.on_key_press)
        key_listener.start()
        
        while not rospy.is_shutdown():
            if self.is_recording and self.sensor_manager.sensors_ready():
                self.record_row()
            rate.sleep()

    
if __name__ == "__main__":
    try: 
        recorder = Recorder()
        recorder.run()
    except KeyboardInterrupt:
        pass