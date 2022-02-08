#!/usr/bin/env python
from collections import namedtuple
from e2e_handover.gripper import ObjDetection, obj_msg_to_enum, open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from e2e_handover.image_ops import prepare_image
from e2e_handover.train.model_double import MultiViewResNet
from enum import Enum
from numpy.core.numeric import NaN
import os
from pynput import keyboard
from e2e_handover.sensor_manager import SensorManager
from sensor_msgs.msg import Joy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
import rospkg
import rospy
from std_msgs.msg import Bool
import torch

# Node to record data and perform inference given a trained model
# Launch node, gripper opens
# Give object to arm (triggered by force threshold)
# Can take object from arm if needed (triggered by force threshold)
# Button to toggle recording and toggle gripper

# Data is stored in 'data/${SESSION_ID}' folder, where SESSION_ID is unique timestamp
# Images are stored in that folder with index and session id as name
# Each session folder has a csv with numerical data and gripper state associated with image name

class GripState(Enum):
    WAITING=1
    GRABBING=2
    HOLDING=3
    RELEASING=4

MODEL_THRESHOLD = 0.5

class HandoverNode():
    def __init__(self):
        rospy.init_node("inference")

        inference_params = rospy.get_param('~inference')
        self.sensor_manager = SensorManager(inference_params)

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) # joystick control
        self.recording_state_sub = rospy.Subscriber('/recorder/is_recording', Bool, self.recording_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.current_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.toggle_key_pressed = False

        self.is_inference_active = False
        self.is_recording = False

        # Create network and load weights
        model_file = inference_params['model_file']
        params = namedtuple("Params", inference_params.keys())(*inference_params.values())
        rospack = rospkg.RosPack()
        package_dir = rospack.get_path('e2e_handover')
        model_path = os.path.join(package_dir, model_file)
        rospy.loginfo(f"Using model: {model_path}")
        if os.path.isfile(model_path):
            self.net = MultiViewResNet(params)
            self.net.load_state_dict(torch.load(model_path))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rospy.loginfo("Using device: " + str(self.device))
            self.net.to(self.device)
            self.net.eval()
        else:
            self.net = None
            rospy.logwarn(f"Unable to load model at {model_path}")

        self.model_output = NaN

    def recording_state_callback(self, msg):
        self.is_recording = msg.data

    def spin_inference(self):
        if self.is_inference_active and self.sensor_manager.sensors_ready() and self.net is not None:
            img_rgb_1 = self.sensor_manager.img_rgb_1[:, :, ::-1]
            img_rgb_1_t = prepare_image(img_rgb_1).unsqueeze_(0).to(self.device)
            forces_t = torch.autograd.Variable(torch.FloatTensor(self.sensor_manager.calib_wrench_array)).unsqueeze_(0).to(self.device)

            output_t = self.net(img_rgb_1_t, forces_t)
            self.model_output = output_t.cpu().detach().numpy()[0][0]

    def gripper_state_callback(self, gripper_input_msg):
        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

    def toggle_gripper(self):
        self.toggle_key_pressed = True

    def joy_callback(self, joy_msg):
        down_pressed = joy_msg.axes[7] == -1
        x_pressed = joy_msg.buttons[0] == 1
        option_pressed = joy_msg.buttons[9] == 1

        if option_pressed:
            self.toggle_inference()

        if down_pressed or x_pressed:
            self.toggle_gripper()

    def toggle_inference(self):
        self.is_inference_active = not self.is_inference_active
        self.model_output = NaN

        if self.is_inference_active:
            self.sensor_manager.activate()
        else:
            self.sensor_manager.deactivate()

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'i': # i key to start/stop inference controlling the gripper
                self.toggle_inference()
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.toggle_gripper()

    def compute_next_state(self, toggle_key_pressed):
        next_state = self.current_state

        if self.current_state == GripState.HOLDING:
            if toggle_key_pressed or self.model_output > MODEL_THRESHOLD:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_state == GripState.WAITING:
            if toggle_key_pressed or self.model_output < MODEL_THRESHOLD:
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

        # initialise the gripper via reset and activate messages
        grip_cmd = reset_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        rospy.sleep(0.1)
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # keyboard input
        key_listener = keyboard.Listener(on_press=self.on_key_press)
        key_listener.start()
        
        while not rospy.is_shutdown():
            self.spin_inference()
            next_state = self.compute_next_state(self.toggle_key_pressed)
            self.toggle_key_pressed = False
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))
                self.current_state = next_state

            rospy.loginfo(f"Rec: {self.is_recording}, infer: {self.is_inference_active}, out: %.4f, {self.obj_det_state}, {self.current_state}", self.model_output)

            rate.sleep()

if __name__ == "__main__":
    try: 
        handover = HandoverNode()
        handover.run()
    except KeyboardInterrupt:
        pass