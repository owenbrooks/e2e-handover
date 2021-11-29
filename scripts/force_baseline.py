#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import WrenchStamped
from math import sqrt
from enum import Enum
# states: holding, grabbing, releasing, waiting
# continually read force/torque values
# transition state if thresholds are reached

class GripState(Enum):
    GRABBING=1
    HOLDING=2
    RELEASING=3
    WAITING=4

GRAB_THRESHOLD = 15 # Newtons
RELEASE_THRESHOLD = 20 # Newtons

class ForceBaselineNode():
    def __init__(self):
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        super(ForceBaselineNode, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("force_baseline")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)

        self.current_state = GripState.WAITING

    def force_callback(self, wrench_msg):
        self.latest_force = norm3(wrench_msg.wrench.force)

    def compute_next_state(force):
        next_state = current_state

        if current_state == GripState.HOLDING:
            if force > RELEASE_THRESHOLD:
                next_state = GripState.RELEASING
        elif current_state == GripState.WAITING:
            if force > GRAB_THRESHOLD:
                next_state = GripState.GRABBING


        return next_state
    
    def run(self):
        rospy.loginfo("Running force thresholding baseline node")

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            next_state = compute_next_state(self.current_state, self.latest_force)
            if next_state != self.current_state:
                print("" + str(self.current_state) + " -> " + str(next_state))

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