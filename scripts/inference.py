#!/usr/bin/env python
import os
import csv

from numpy.core.numeric import NaN
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy, Image
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from enum import Enum
from robot_control.gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from datetime import datetime
from cv_bridge import CvBridge, CvBridgeError
import cv2
from pynput import keyboard
from robot_control import model
import torch
from robot_control.image_ops import prepare_image

# Node to record data and perform inference given a trained model
# Launch node, gripper opens
# Give object to arm (triggered by force threshold)
# Can take object from arm if needed (triggered by force threshold)
# Button to toggle recording
# Button to toggle gripper

# Data is stored in 'data/${SESSION_ID}' folder, where SESSION_ID is unique timestamp
# Images are stored in that folder with index and session id as name
# Each session folder has a csv with numerical data and gripper state associated with image name

class GripState(Enum):
    WAITING=1
    GRABBING=2
    HOLDING=3
    RELEASING=4

class ObjDetection(Enum):
    IN_MOTION=0
    OPENING_STOPPED=1
    CLOSING_STOPPED=2
    FINISHED_MOTION=3
    GRIPPER_OFFLINE=4

# Used to map gripper state integer to our enum values
obj_msg_to_enum = {
    0: ObjDetection.IN_MOTION, 
    1: ObjDetection.OPENING_STOPPED, 
    2: ObjDetection.CLOSING_STOPPED, 
    3: ObjDetection.FINISHED_MOTION
}

GRAB_THRESHOLD_FORCE = 8 # Newtons
RELEASE_THRESHOLD_FORCE = 8 # Newtons
MODEL_THRESHOLD = 0.5

class InferenceNode():
    def __init__(self):
        rospy.init_node("inference")

        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) # joystick control
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.cv_bridge = CvBridge() # for converting ROS image messages to OpenCV images

        self.current_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.abs_z_force = 0.0
        self.toggle_key_pressed = False

        self.is_recording = False
        self.is_inference_active = False
        self.session_id = ""

        self.session_image_count = 0

        self.wrench_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Create folder for storing recorded images and the csv with numerical/annotation data
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../data')
        data_dir_exists = os.path.exists(data_dir)
        if not data_dir_exists:
            os.makedirs(data_dir)

        # Create network and load weights
        model_name = rospy.get_param("model_name", default='2021-12-09-04:56:05.pt')
        self.net = model.ResNet()
        model_path = os.path.join(current_dirname, '../models', model_name)
        self.net.load_state_dict(torch.load(model_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo("Using device: " + str(self.device))
        self.net.to(self.device)

        self.model_output = NaN

    def force_callback(self, wrench_msg):
        self.abs_z_force = abs(wrench_msg.wrench.force.z)

        self.wrench_array = [wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z, 
            wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z]
    
    def gripper_state_callback(self, gripper_input_msg):
        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

    def start_recording(self):
        if self.is_recording:
            rospy.loginfo("Already recording. Session: " + self.session_id)
        else:
            self.is_recording = True
            self.session_image_count = 0
            # Create folder for storing recorded images and the session csv
            self.session_id = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            current_dirname = os.path.dirname(__file__)
            session_dir = os.path.join(current_dirname, '../data', self.session_id, 'images')
            os.makedirs(session_dir)
            # Create csv file for recording numerical data and annotation in the current session
            fname = os.path.join(session_dir, self.session_id + '.csv')
            with open(fname, 'w+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                header = ['image_id', 'gripper_is_open', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
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

    def joy_callback(self, joy_msg):
        share_pressed = joy_msg.buttons[8]
        down_pressed = joy_msg.axes[7] == -1
        x_pressed = joy_msg.buttons[0] == 1

        if share_pressed:
            self.toggle_recording()

        if down_pressed or x_pressed:
            self.toggle_gripper()

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'r': # r key to start/stop recording
                self.toggle_recording()
            if char == 'i': # i key to start/stop inference controlling the gripper
                self.is_inference_active = not self.is_inference_active
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r: # space bar to open/close gripper
                self.toggle_gripper()

    def image_callback(self, image_msg):
        if self.is_recording:
            gripper_is_open = self.current_state == GripState.RELEASING or self.current_state == GripState.WAITING


            # Save image as png
            current_dirname = os.path.dirname(__file__)
            image_name = str(self.session_image_count) + '.png'
            self.session_image_count += 1
            image_path = os.path.join(current_dirname, '../data', self.session_id, 'images', image_name)
            try:
                cv2_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
            else:
                cv2.imwrite(image_path, cv2_img)

            # Append numerical data and annotation to the session csv
            csv_path = os.path.join(current_dirname, '../data', self.session_id, self.session_id + '.csv')
            with open(csv_path, 'a+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ')
                datawriter.writerow([image_name, gripper_is_open] + self.wrench_array)
        
        if self.is_inference_active:
            gripper_is_open = self.current_state == GripState.RELEASING or self.current_state == GripState.WAITING
            
            img_cv2 = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            img_t = prepare_image(img_cv2).unsqueeze_(0).to(self.device)
            forces_t = torch.autograd.Variable(torch.FloatTensor(self.wrench_array)).unsqueeze_(0).to(self.device)

            # forward + backward + optimize
            output_t = self.net(img_t, forces_t)
            self.model_output = output_t.cpu().detach().numpy()[0]

    def compute_next_state(self, force, toggle_key_pressed):
        next_state = self.current_state

        if self.current_state == GripState.HOLDING:
            if force > RELEASE_THRESHOLD_FORCE or toggle_key_pressed or self.model_output > MODEL_THRESHOLD:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_state == GripState.WAITING:
            if force > GRAB_THRESHOLD_FORCE or toggle_key_pressed or self.model_output < MODEL_THRESHOLD:
                next_state = GripState.GRABBING
                # close gripper
                grip_cmd = close_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_state == GripState.GRABBING:
            if self.obj_det_state == ObjDetection.CLOSING_STOPPED:
                next_state = GripState.HOLDING
            elif self.obj_det_state == ObjDetection.FINISHED_MOTION:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_state == GripState.RELEASING:
            if self.obj_det_state == ObjDetection.FINISHED_MOTION:
                next_state = GripState.WAITING

        return next_state
    
    def run(self):
        rospy.loginfo("Running inference node")
        rate = rospy.Rate(10)

        while self.obj_det_state == ObjDetection.GRIPPER_OFFLINE and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rate.sleep()

        # initialise the gripper
        grip_cmd = reset_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # open the gripper to start
        grip_cmd = open_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # keyboard input
        key_listener = keyboard.Listener(
            on_press=self.on_key_press)
        key_listener.start()
        
        while not rospy.is_shutdown():
            next_state = self.compute_next_state(self.abs_z_force, self.toggle_key_pressed)
            self.toggle_key_pressed = False
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))
                self.current_state = next_state

            rospy.loginfo("Rec: %s, infer: %s, out: %.3f, f_z: %.2f, %s, %s", self.is_recording, self.is_inference_active, self.model_output, self.abs_z_force, self.obj_det_state, self.current_state)

            rate.sleep()

if __name__ == "__main__":
    try: 
        inference = InferenceNode()
        inference.run()
    except KeyboardInterrupt:
        pass