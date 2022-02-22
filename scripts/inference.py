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
from std_srvs.srv import Trigger
import sys
import torch

# Node to perform inference given a trained model
# Launch node, gripper opens
# Button to toggle gripper

# Data is stored in 'data/${SESSION_ID}' folder, where SESSION_ID is unique timestamp
# Images are stored in that folder with index and session id as name
# Each session folder has a csv with numerical data and gripper state associated with image name

class GripState(Enum):
    WAITING=1
    GRABBING=2
    HOLDING=3
    RELEASING=4

class HandoverState(Enum):
    GIVING=1
    RECEIVING=2

class MotionState(Enum):
    INITIALISING=0
    RETRACTED=1
    REACHING=2
    EXTENDED=3
    RETURNING=4

MODEL_THRESHOLD = 0.5

class HandoverNode():
    def __init__(self):
        rospy.init_node("inference")

        inference_params = rospy.get_param('~inference')
        self.in_simulation = rospy.get_param('~in_simulation')
        self.sensor_manager = SensorManager(inference_params['giving']) # TODO feed in a combination of both to get all sensors
        self.motion_state = MotionState.INITIALISING

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) # joystick control
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.current_gripper_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.handover_state = HandoverState.RECEIVING
        self.toggle_key_pressed = False

        self.mover_setup = rospy.ServiceProxy('/mover/setup', Trigger)
        self.mover_reach = rospy.ServiceProxy('/mover/reach', Trigger)
        self.mover_retract = rospy.ServiceProxy('/mover/retract', Trigger)
        self.last_retracted = rospy.get_time()
        self.wait_time_threshold = 2.0 # time to wait before reaching out after having retracted. in seconds

        self.is_inference_active = False

        # Determine model paths
        self.model_paths = {}
        self.params = {}
        for action_string in ['giving', 'receiving']:
            model_file = inference_params[action_string]['model_file']
            self.params[action_string] = namedtuple("Params", inference_params[action_string].keys())(*inference_params[action_string].values())
            rospack = rospkg.RosPack()
            package_dir = rospack.get_path('e2e_handover')
            self.model_paths[action_string] = os.path.join(package_dir, model_file)
            rospy.loginfo(f"Using model: {self.model_paths[action_string]} for {action_string}")

        self.model_output = NaN

    def spin_inference(self):
        # TODO: feed correct sensor data to model according to handover_params.yaml
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

    def toggle_inference(self, set_on: bool=None):
        if set_on is None: # just toggle it
            self.is_inference_active = not self.is_inference_active
        else:
            self.is_inference_active = set_on

        self.model_output = NaN

        if self.is_inference_active:
            self.sensor_manager.activate()
        else:
            self.sensor_manager.deactivate()

        if set_on:
            self.load_model()

    def load_model(self):
        action_string = 'giving' if self.handover_state == HandoverState.GIVING else 'receiving'
        model_path = self.model_paths[action_string]
        if os.path.isfile(model_path):
            self.net = MultiViewResNet(self.params[action_string])
            self.net.load_state_dict(torch.load(model_path))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rospy.loginfo("Using device: " + str(self.device))
            self.net.to(self.device)
            self.net.eval()
        else:
            self.net = None
            rospy.logerr(f"Unable to load model at {model_path}")

    def on_key_press(self, key):
        try:
            char = key.char
            if char == 'i': # i key to start/stop inference controlling the gripper
                self.toggle_inference()
        except AttributeError: # special keys (ctrl, alt, etc.) will cause this exception
            if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
                self.toggle_gripper()

    def compute_next_gripper_state(self, toggle_key_pressed):
        next_state = self.current_gripper_state

        if self.current_gripper_state == GripState.HOLDING:
            if toggle_key_pressed or self.model_output > MODEL_THRESHOLD or self.in_simulation:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
                if self.params['giving'].use_tactile or self.params['receiving'].use_tactile:
                    self.sensor_manager.contactile_bias_srv()
        elif self.current_gripper_state == GripState.WAITING:
            if toggle_key_pressed or self.model_output < MODEL_THRESHOLD or self.in_simulation:
                next_state = GripState.GRABBING
                # close gripper
                grip_cmd = close_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
        elif self.current_gripper_state == GripState.GRABBING:
            if self.obj_det_state == ObjDetection.CLOSING_STOPPED or self.in_simulation:
                next_state = GripState.HOLDING
            elif self.obj_det_state == ObjDetection.FINISHED_MOTION:
                next_state = GripState.RELEASING
                # open gripper
                grip_cmd = open_gripper_msg()
                self.gripper_pub.publish(grip_cmd)
                if self.params['giving'].use_tactile or self.params['receiving'].use_tactile:
                    self.sensor_manager.contactile_bias_srv()
        elif self.current_gripper_state == GripState.RELEASING:
            if self.obj_det_state == ObjDetection.FINISHED_MOTION or self.in_simulation:
                next_state = GripState.WAITING

        return next_state

    def transition_state(self, curr_handover, next_handover, curr_mover, next_mover):
        self.handover_state = next_handover
        
        if curr_mover == MotionState.RETRACTED and next_mover == MotionState.REACHING:
            self.motion_state = MotionState.REACHING
            response = self.mover_reach()
            if not response.success:
                rospy.logerr("Movement unsuccessful")
                sys.exit()
            self.motion_state = MotionState.EXTENDED
            self.toggle_inference(set_on=True)
        elif (curr_mover == MotionState.EXTENDED or curr_mover == MotionState.INITIALISING) and next_mover == MotionState.RETURNING:
            self.toggle_inference(set_on=False)
            self.motion_state = MotionState.RETURNING
            response = self.mover_retract()
            if not response.success:
                rospy.logerr("Movement unsuccessful")
                sys.exit()
            self.motion_state = MotionState.RETRACTED

            # Switch from giving to receiving or vice versa
            if curr_handover == HandoverState.GIVING:
                self.handover_state = HandoverState.RECEIVING
            else:
                self.handover_state = HandoverState.GIVING

            self.last_retracted = rospy.get_time() # (re)start timing of retraction
    
    def run(self):
        rospy.loginfo("Running inference node")
        rate = rospy.Rate(10)

        # Wait to start until robot is sent to position and gripper is connected
        rospy.loginfo("Waiting for mover to be ready to send robot to position...")
        rospy.wait_for_service('/mover/setup')
        response = self.mover_setup()
        if response.success:
            self.last_retracted = rospy.get_time()
            self.motion_state = MotionState.RETRACTED
        else:
            rospy.logerr("Movement unsuccessful")
            sys.exit()
        # self.transition_state(self.handover_state, self.handover_state, self.motion_state, MotionState.RETURNING)
        if not self.in_simulation: # don't wait for gripper if simulated, since there isn't a gripper in gazebo
            while self.obj_det_state == ObjDetection.GRIPPER_OFFLINE and not rospy.is_shutdown():
                rospy.loginfo("Waiting for gripper to connect...")
                rate.sleep()

        # Initialise the gripper via reset and activate messages
        grip_cmd = reset_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        rospy.sleep(0.1)
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # Setup keyboard input
        key_listener = keyboard.Listener(on_press=self.on_key_press)
        key_listener.start()
        
        while not rospy.is_shutdown():
            self.spin_inference()

            # Spin state machine that takes care of the actual gripper
            next_gripper_state = self.compute_next_gripper_state(self.toggle_key_pressed)
            self.toggle_key_pressed = False
            if next_gripper_state != self.current_gripper_state:
                print("" + str(self.current_gripper_state) + " -> " + str(next_gripper_state))
                self.current_gripper_state = next_gripper_state

            # Spin state machine that determines arm motion and inference model to use
            retracted_time = rospy.get_time() - self.last_retracted
            next_handover_state, next_motion_state = compute_handover_and_motion_states(self.handover_state, self.motion_state, next_gripper_state, retracted_time, self.wait_time_threshold)
            rospy.loginfo(f"infer: {self.is_inference_active}, out: {self.model_output: .4f}, {self.obj_det_state}, {self.current_gripper_state}, {self.motion_state}, {self.handover_state}, ret_time: {retracted_time}", )
            self.transition_state(self.handover_state, next_handover_state, self.motion_state, next_motion_state)
            
            rate.sleep()


def compute_handover_and_motion_states(curr_handover, curr_motion, next_gripper, retracted_time, wait_time_threshold):
    next_motion = curr_motion
    next_handover = curr_handover

    if curr_motion == MotionState.RETRACTED:
        if retracted_time > wait_time_threshold:
            next_motion = MotionState.REACHING
    elif curr_motion == MotionState.EXTENDED:
        received = curr_handover == HandoverState.RECEIVING and next_gripper == GripState.HOLDING
        released = curr_handover == HandoverState.GIVING and next_gripper == GripState.WAITING
        if received or released:
            next_motion = MotionState.RETURNING

    return next_handover, next_motion

if __name__ == "__main__":
    try: 
        handover = HandoverNode()
        handover.run()
    except KeyboardInterrupt:
        pass