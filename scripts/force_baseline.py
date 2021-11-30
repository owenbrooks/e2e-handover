#!/usr/bin/env python
import sys
import rospy
# import moveit_commander
# import moveit_msgs.msg
from geometry_msgs.msg import WrenchStamped
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from math import sqrt
from enum import Enum
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg
# states: holding, grabbing, releasing, waiting
# continually read force/torque values
# transition state if thresholds are reached

class GripState(Enum):
    GRABBING=1
    HOLDING=2
    RELEASING=3
    WAITING=4

class ObjDetection(Enum):
    IN_MOTION=0
    OPENING_STOPPED=1
    CLOSING_STOPPED=2
    FINISHED_MOTION=3
    GRIPPER_OFFLINE=4

obj_msg_to_enum = {0: ObjDetection.IN_MOTION, 1: ObjDetection.OPENING_STOPPED, 2: ObjDetection.CLOSING_STOPPED, 3: ObjDetection.FINISHED_MOTION}

GRAB_THRESHOLD = 8 # Newtons
RELEASE_THRESHOLD = 8 # Newtons

class ForceBaselineNode():
    def __init__(self):
        # self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
        #                                        moveit_msgs.msg.DisplayTrajectory,
        #                                        queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        # moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("force_baseline")

        # self.robot = moveit_commander.RobotCommander()
        # self.scene = moveit_commander.PlanningSceneInterface()
        # self.group_name = "manipulator"
        # self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.current_state = GripState.WAITING
        self.obj_det_state = ObjDetection.GRIPPER_OFFLINE
        self.latest_force = 0.0

    def force_callback(self, wrench_msg):
        self.latest_force = abs(wrench_msg.wrench.force.z)
    
    def gripper_state_callback(self, gripper_input_msg):
        self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]

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
        rospy.loginfo("Running force thresholding baseline node")
        rate = rospy.Rate(10)

        while self.obj_det_state == ObjDetection.GRIPPER_OFFLINE and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rate.sleep()

        # initialise the gripper
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # open the gripper to start
        grip_cmd = open_gripper_msg()
        self.gripper_pub.publish(grip_cmd)
        
        while not rospy.is_shutdown():
            next_state = self.compute_next_state(self.latest_force)
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))
                self.current_state = next_state

            rospy.loginfo("Force: %f, object_status: %s, state: %s", self.latest_force, self.obj_det_state, self.current_state)

            rate.sleep()

def norm3(vector3):
    return sqrt(vector3.x**2 + vector3.y**2 + vector3.z**2)

def gripper_cmd(action_request, goto, auto_release, auto_release_dir, pos_request, speed, force):
    pass

if __name__ == "__main__":
    try: 
        force_baseline = ForceBaselineNode()
        force_baseline.run()
    except KeyboardInterrupt:
        pass