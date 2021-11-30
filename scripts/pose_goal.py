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

class PoseGoal(object):
    def __init__(self):
        self.move_back_srv = rospy.Service("switch_pos", Trigger, self.switch_pos)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        super(PoseGoal, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("home_position")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.state_index = 0

        home_pose_1 = [0.075, 0.449, 0.777, 0.702, -0.261, 0.643, -0.161]
        home_pose_2 =  [-0.383, 0.044, 0.702, -0.570, -0.421, -0.418, 0.569]

        self.states = [home_pose_1, home_pose_2]

        # set up table object
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "base_link"
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.z = -1.0 # slightly above the end effector
        table_name = "table"
        self.scene.add_box(table_name, table_pose, size=(1.0, 1.0, 1.0))

    def switch_pos(self, msg):
        self.state_index = 0 if self.state_index == 1 else 1
        self.move_to_pose(self.state_index)
        rospy.loginfo("Moving to state " + str(self.state_index))
        return TriggerResponse(success=True, message="")

    def move_to_pose(self, state_index):
        # plan = self.move_group.go(self.states[state_index], wait=True)
        print(self.state_index)
        self.move_group.set_pose_target(self.states[state_index])
        plan = self.move_group.plan()
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

        run_flag = raw_input("Valid Trajectory? [y to run]:")

        if run_flag =="y":
            self.move_group.execute(plan, wait=True)

        self.move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.move_group.clear_pose_targets()
    
    def run(self):
        rospy.loginfo("Running pose goal node")

        rate = rospy.Rate(10)

        self.move_to_pose(self.state_index)

        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try: 
        test_control = PoseGoal()
        test_control.run()
    except KeyboardInterrupt:
        pass