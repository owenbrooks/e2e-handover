#!/usr/bin/env python
import sys
import os
import csv
import rospy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Joy, Image
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from math import sqrt
from enum import Enum
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg
from datetime import datetime
from uuid import uuid4
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import cv2

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
    RELEASING=3

class ObjDetection(Enum):
    IN_MOTION=0
    OPENING_STOPPED=1
    CLOSING_STOPPED=2
    FINISHED_MOTION=3
    GRIPPER_OFFLINE=4

obj_msg_to_enum = {0: ObjDetection.IN_MOTION, 1: ObjDetection.OPENING_STOPPED, 2: ObjDetection.CLOSING_STOPPED, 3: ObjDetection.FINISHED_MOTION}

GRAB_THRESHOLD = 8 # Newtons
RELEASE_THRESHOLD = 8 # Newtons

class RecorderNode():
    def __init__(self):
        rospy.init_node("recorder")

        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        joy_sub = message_filters.Subscriber('joy', Joy)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, joy_sub], 10, 0.1) # TODO: how to choose slop?
        ts.registerCallback(self.image_and_input_callback)

        self.cv_bridge = CvBridge() # for converting ROS image messages to OpenCV images

        self.current_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.abs_z_force = 0.0

        self.is_recording = False
        self.session_id = ""

        self.session_image_count = 0

        # Create folder for storing recorded images and the csv with numerical/annotation data
        current_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(current_dirname, '../data')
        data_dir_exists = os.path.exists(data_dir)
        if not data_dir_exists:
            os.makedirs(data_dir)

    def force_callback(self, wrench_msg):
        self.abs_z_force = abs(wrench_msg.wrench.force.z)
    
    def gripper_state_callback(self, gripper_input_msg):
        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

    def start_recording(self):
        if self.is_recording:
            rospy.loginfo("Already recording. Session: " + self.session_id)
        else:
            self.is_recording = True
            self.session_image_count = 0
            # Create folder for storing recorded images and the session csv
            self.session_id = datetime.now().strftime('%Y-%m-%d-%H:%M:%S-') + str(uuid4())
            current_dirname = os.path.dirname(__file__)
            session_dir = os.path.join(current_dirname, '../data', self.session_id)
            os.makedirs(session_dir)
            # Create csv file for recording numerical data and annotation in the current session
            current_dirname = os.path.dirname(__file__)
            fname = os.path.join(current_dirname, '../data', self.session_id, self.session_id + '.csv')
            with open(fname, 'w+') as csvfile:
                datawriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                header = ['image_id', 'abs_z_force', 'gripper_is_open']
                datawriter.writerow(header)

            rospy.loginfo("Started recording. Session: " + self.session_id)

    def stop_recording(self):
        if not self.is_recording:
            rospy.loginfo("Hadn't yet started recording.")
        else:
            self.is_recording = False
            rospy.loginfo("Finished recording. Session: " + self.session_id)
            rospy.loginfo("Recorded " + str(self.session_image_count) + " images")

    def image_and_input_callback(self, image_msg, joy_msg):
        gripper_is_open = self.current_state == GripState.RELEASING or self.current_state == GripState.WAITING

        current_dirname = os.path.dirname(__file__)
        
        # Save image as png
        image_name = str(self.session_image_count) + '_' + self.session_id + '.png'
        self.session_image_count += 1
        image_path = os.path.join(current_dirname, '../data', self.session_id, image_name)
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
            datawriter.writerow([image_name, self.abs_z_force, gripper_is_open])

    def compute_next_state(self, force):
        next_state = self.current_state

        if self.current_state == GripState.HOLDING:
            if force > RELEASE_THRESHOLD:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_state == GripState.WAITING:
            if force > GRAB_THRESHOLD:
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
        rospy.loginfo("Running recorder node")
        rate = rospy.Rate(10)

        # while self.obj_det_state == ObjDetection.GRIPPER_OFFLINE and not rospy.is_shutdown():
        #     rospy.loginfo("Waiting for gripper to connect")
        #     rate.sleep()

        # initialise the gripper
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # open the gripper to start
        grip_cmd = open_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        self.start_recording()
        
        while not rospy.is_shutdown():
            next_state = self.compute_next_state(self.abs_z_force)
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))
                self.current_state = next_state

            rospy.loginfo("Recording: %s, f_z: %.2f, obj: %s, state: %s", self.is_recording, self.abs_z_force, self.obj_det_state, self.current_state)

            rate.sleep()

if __name__ == "__main__":
    try: 
        recorder = RecorderNode()
        recorder.run()
    except KeyboardInterrupt:
        pass