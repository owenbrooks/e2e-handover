#!/usr/bin/env python3
import csv
import cv2
from datetime import datetime
from e2e_handover import tactile
from e2e_handover.gripper import ObjDetection, obj_msg_to_enum, open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from e2e_handover.sensor_manager import SensorManager
from inference import GripState
from geometry_msgs.msg import Twist
import os
from pynput import keyboard
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
import rospkg
import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

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
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) # joystick control

        self.recording_pub = rospy.Publisher('~is_recording', Bool, queue_size=10)

        self.is_recording = False
        self.session_id = ""
        self.row_count = 0

        self.twist_array = 6*[0.0]
        self.filtered_twist_array = 6*[0.0]

        self.gripper_is_open = True
        self.gripper_pos = 0
        self.desired_open = True
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE

        self.toggle_key_pressed = False

        self.recording_params = rospy.get_param('~recording')
        self.sensor_manager = SensorManager(self.recording_params)

        self.image_classes = []
        for key in ['rgb_1', 'rgb_2', 'depth_1', 'depth_2']:
            if self.recording_params['use_' + key]:
                self.image_classes.append(key)

        # Create folder for storing recorded images and the csv with numerical/annotation data
        rospack = rospkg.RosPack()
        package_dir = rospack.get_path('e2e_handover')
        self.data_dir = os.path.join(package_dir, 'data')
        data_dir_exists = os.path.exists(self.data_dir)
        if not data_dir_exists:
            os.makedirs(self.data_dir)

    def gripper_state_callback(self, gripper_input_msg):
        # print(gripper_input_msg)
        position_requested = gripper_input_msg.gPR
        position_actual = gripper_input_msg.gPO
        self.gripper_pos = position_actual
        # should have value 0x00 for open, 0xFF for closed
        if position_requested == 0:
            self.gripper_is_open = True
        else:
            self.gripper_is_open = False

        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

    def twist_callback(self, twist_msg):
        self.twist_array = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]
    
    def filtered_twist_callback(self, twist_msg):
        self.filtered_twist_array = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z, twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]

    def save_image(self, image, count, identifier):
        # Save image as png
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        image_name = f"{count}_{timestamp}.png"
        rel_path = os.path.join(identifier, image_name)
        image_path = os.path.join(self.data_dir, self.session_id, rel_path)
        cv2.imwrite(image_path, image)

        return rel_path

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

            for image_class in self.image_classes:
                image_dir = os.path.join(session_dir, 'image_' + image_class)
                os.makedirs(image_dir)
        
            # Create csv file for recording numerical data and annotation in the current session
            fname = os.path.join(self.data_dir, self.session_id, self.session_id + '.csv')
            with open(fname, 'w+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                
                wrench_header = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']
                twist_header = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
                filtered_twist_header = ['vx_filt', 'vy_filt', 'vz_filt', 'wx_filt', 'wy_filt', 'wz_filt']
                
                header = ['gripper_is_open', 'desired_open'] + twist_header + filtered_twist_header
                
                for image_class in self.image_classes:
                    header.append('image_' + image_class)

                if self.recording_params['use_force']:
                    header += wrench_header

                if self.recording_params['use_tactile']:
                    header += tactile.papillarray_keys
            
                datawriter.writerow(header)

            rospy.loginfo("Started recording. Session: " + self.session_id)

    def record_row(self):
        # Append numerical data and annotation to the session csv
        csv_path = os.path.join(self.data_dir, self.session_id, self.session_id + '.csv')
        with open(csv_path, 'a+') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=' ')
            row = [self.gripper_is_open, self.desired_open] + self.twist_array + self.filtered_twist_array

            image_data = {
                'rgb_1': self.sensor_manager.img_rgb_1,
                'rgb_2': self.sensor_manager.img_rgb_2,
                'depth_1': self.sensor_manager.img_depth_1,
                'depth_2': self.sensor_manager.img_depth_2,
            }
            for image_class in self.image_classes:
                rel_path = self.save_image(image_data[image_class], self.row_count, 'image_' + image_class)
                row.append(rel_path)

            if self.recording_params['use_force']:
                row += self.sensor_manager.raw_wrench_reading.tolist()

            if self.recording_params['use_tactile']:
                row += self.sensor_manager.tactile_1_readings
                row += self.sensor_manager.tactile_2_readings

            datawriter.writerow(row)
            self.row_count += 1

    def stop_recording(self):
        if not self.is_recording:
            rospy.loginfo("Hadn't yet started recording.")
        else:
            self.is_recording = False
            # self.sensor_manager.deactivate()
            rospy.loginfo(f"Finished recording. Session: {self.session_id}")
            rospy.loginfo(f"Recorded {self.row_count + 1} frames")

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_gripper(self):
        self.toggle_key_pressed = True

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'r': # r key to start/stop recording
                self.toggle_recording()
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.toggle_gripper()

    def joy_callback(self, joy_msg):
        share_pressed = joy_msg.buttons[8]
        triangle_pressed = joy_msg.buttons[2]
        up_pressed= joy_msg.axes[7]
        down_pressed = joy_msg.axes[7] == -1
        x_pressed = joy_msg.buttons[0] == 1

        if share_pressed:
            self.toggle_recording()

        if down_pressed or x_pressed:
            self.toggle_gripper()
        
        # Pressing either of these buttons indicates that the robot should learn to associate this with the closed state
        self.desired_open = not triangle_pressed and not up_pressed

    def open_gripper(self):
        grip_cmd = open_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        # self.sensor_manager.contactile_bias_srv()
    
    def close_gripper(self):
        grip_cmd = close_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

    def compute_next_state(self, current_state, toggle_key_pressed):
        next_state = current_state

        gripper_closed = self.gripper_pos > 20 # used since it takes time for the ObjDet state to change from FINISHED_MOTION to IN_MOTION

        if current_state == GripState.HOLDING:
            if toggle_key_pressed:
                next_state = GripState.RELEASING
                self.open_gripper()
        elif current_state == GripState.WAITING:
            if toggle_key_pressed:
                next_state = GripState.GRABBING
                self.close_gripper()
        elif current_state == GripState.GRABBING:
            if self.obj_det_state == ObjDetection.CLOSING_STOPPED:
                next_state = GripState.HOLDING
            elif self.obj_det_state == ObjDetection.FINISHED_MOTION and gripper_closed:
                next_state = GripState.RELEASING
                self.open_gripper()
        elif current_state == GripState.RELEASING:
            if self.obj_det_state == ObjDetection.FINISHED_MOTION:
                next_state = GripState.WAITING

        return next_state

    def run(self):
        rate = rospy.Rate(30)

        while self.obj_det_state == ObjDetection.GRIPPER_OFFLINE and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rate.sleep()

        # initialise the gripper via reset and activate messages
        grip_cmd = reset_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        rospy.sleep(0.1)
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # keyboard input
        key_listener = keyboard.Listener(on_press=self.on_key_press)
        key_listener.start()

        rospy.loginfo("Recorder node up.")

        current_state = GripState.WAITING
        
        while not rospy.is_shutdown():
            if self.is_recording and self.sensor_manager.sensors_ready():
                self.record_row()

            next_state = self.compute_next_state(current_state, self.toggle_key_pressed)
            # rospy.loginfo(f"{self.gripper_pos} {current_state}, {self.obj_det_state}, {self.toggle_key_pressed}")
            self.toggle_key_pressed = False
            if next_state != current_state:
                print("" + str(current_state) + " -> " + str(next_state))
                current_state = next_state

            rate.sleep()

    
if __name__ == "__main__":
    try: 
        recorder = Recorder()
        recorder.run()
    except KeyboardInterrupt:
        pass