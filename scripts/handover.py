#!/usr/bin/env python
from e2e_handover.gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from e2e_handover.train import model
from enum import Enum
from numpy.core.numeric import NaN
import os
from pynput import keyboard
from sensor_msgs.msg import Joy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
import rospkg
import rospy
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

GRAB_THRESHOLD_FORCE = 50 # Newtons
RELEASE_THRESHOLD_FORCE = 50 # Newtons
MODEL_THRESHOLD = 0.5

class InferenceNode():
    def __init__(self):
        rospy.init_node("inference")

        inference_params = rospy.get_param('~inference')

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) # joystick control
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.current_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.toggle_key_pressed = False

        self.is_inference_active = False

        # Create network and load weights
        model_file = inference_params['model_file']
        rospack = rospkg.RosPack()
        package_dir = rospack.get_path('e2e_handover')
        model_path = os.path.join(package_dir, model_file)
        rospy.loginfo(f"Using model: {model_path}")
        if os.path.isfile(model_path):
            self.net = model.ResNet()
            self.net.load_state_dict(torch.load(model_path))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rospy.loginfo("Using device: " + str(self.device))
            self.net.to(self.device)
            self.net.eval()
        else:
            self.net = None
            rospy.logwarn(f"Unable to load model at {model_path}")

        self.model_output = NaN

    def perform_inference(self):
        if self.is_inference_active and self.net is not None:
            img = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            img = img[:, :, ::-1]
            img_t = prepare_image(img).unsqueeze_(0).to(self.device)
            forces_t = torch.autograd.Variable(torch.FloatTensor(self.calib_wrench_array)).unsqueeze_(0).to(self.device)

            # forward + backward + optimize
            output_t = self.net(img_t, forces_t)
            self.model_output = output_t.cpu().detach().numpy()[0][0]

    def gripper_state_callback(self, gripper_input_msg):
        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

    def toggle_gripper(self):
        self.toggle_key_pressed = True

    def joy_callback(self, joy_msg):
        share_pressed = joy_msg.buttons[8]
        down_pressed = joy_msg.axes[7] == -1
        x_pressed = joy_msg.buttons[0] == 1

        if share_pressed:
            self.recorder.toggle_recording()

        if down_pressed or x_pressed:
            self.toggle_gripper()

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'r': # r key to start/stop recording
                self.recorder.toggle_recording()
            if char == 'i': # i key to start/stop inference controlling the gripper
                self.is_inference_active = not self.is_inference_active
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.toggle_gripper()

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

        # initialise the gripper via reset and activate messages
        grip_cmd = reset_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        rospy.sleep(0.1)
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # keyboard input
        key_listener = keyboard.Listener(
            on_press=self.on_key_press)
        key_listener.start()
        
        while not rospy.is_shutdown():
            next_state = self.compute_next_state(self.abs_z_force, self.toggle_key_pressed)
            # TODO: run inference here
            self.toggle_key_pressed = False
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))
                self.current_state = next_state

            rospy.loginfo("Rec: %s, infer: %s, out: %.4f, f_z: %.2f, %s, %s, tactile: %s", self.recorder.is_recording, self.is_inference_active, self.model_output, self.abs_z_force, self.obj_det_state, self.current_state, use_tactile)

            rate.sleep()

if __name__ == "__main__":
    try: 
        inference = InferenceNode()
        inference.run()
    except KeyboardInterrupt:
        pass