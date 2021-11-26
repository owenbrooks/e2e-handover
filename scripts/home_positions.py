#!/usr/bin/env python
# ROS node that switches between two home positions
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

class HomePosNode(object):
    def __init__(self):
        self.move_back_srv = rospy.Service("switch_pos", Trigger, self.switch_pos)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        super(HomePosNode, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("home_position")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.state_index = 0

        home_pos_1 = [-0.037211508523346204, -0.14613448707173848, 0.00042289139595208525, 0.0618242143312564, 4.002682490433784e-05, 0.1302630262299358]
        home_pos_2 = [-0.31200551281224165, -2.1927985112258135, -0.000551303369098477, 0.8985175465372821, -9.769528357406188e-05, -1.255454304191412]
        self.states = [home_pos_1, home_pos_2]

    def switch_pos(self, msg):
        self.state_index = 0 if self.state_index == 1 else 1
        self.move_to_state(self.state_index)
        rospy.loginfo("Moving to state " + str(self.state_index))
        return TriggerResponse(success=True, message="")

    def move_to_state(self, state_index):
        # plan = self.move_group.go(self.states[state_index], wait=True)
        print(self.state_index)
        plan = self.move_group.plan(self.states[state_index])
        self.move_group.execute(plan, wait=True)

        next_index = 0 if self.state_index == 1 else 1
        next_plan = self.move_group.plan(self.states[next_index])
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(next_plan)
        self.display_trajectory_publisher.publish(display_trajectory)
    
    def run(self):
        rospy.loginfo("Running position node")

        rate = rospy.Rate(10)

        self.move_to_state(self.state_index)

        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try: 
        test_control = HomePosNode()
        test_control.run()
    except KeyboardInterrupt:
        pass