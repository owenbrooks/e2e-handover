#!/usr/bin/env python
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from moveit_commander.conversions import pose_to_list

class TestControlNode(object):
    def __init__(self):
        self.move_back_srv = rospy.Service("move_back", Trigger, self.move_back)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        super(TestControlNode, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("test_control")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

    def move_back(self, msg):
        move_distance = 0.10
        rospy.loginfo("Moving back %f", move_distance)
        self.move_group.shift_pose_target(axis=0, value=-move_distance) # move back in the x direction
        plan_successful = self.move_group.go(wait=True) # execute the plan
        self.move_group.stop() # Calling `stop()` ensures that there is no residual movement
        self.move_group.clear_pose_targets()
        return TriggerResponse(success=plan_successful, message="")
    
    def run(self):
        rospy.loginfo("Running test control node")

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try: 
        test_control = TestControlNode()
        test_control.run()
    except KeyboardInterrupt:
        pass