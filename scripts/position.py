#!/usr/bin/env python
# ROS node to move the robot to a pre-defined position
import sys
import rospy
import moveit_commander
from std_srvs.srv import Trigger, TriggerResponse

class PositionNode(object):
    def __init__(self):
        self.execute_plan_srv = rospy.Service("execute_plan", Trigger, self.execute_plan)
        self.display_plan_srv = rospy.Service("display_plan", Trigger, self.display_plan)
        # self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
        #                                        moveit_msgs.msg.DisplayTrajectory,
        #                                        queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        super(PositionNode, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("position")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.start_position = [-0.037211508523346204, -0.14613448707173848, 0.00042289139595208525, 
            0.0618242143312564, 4.002682490433784e-05, 0.1302630262299358] # joint angles in radians

        self.in_position = False

    def execute_plan(self):
        if self.plan is not None:
            print(f"Moving to {self.start_position}")
            self.move_group.execute(self.plan, wait=True)
            return TriggerResponse(success=True, message="")
        else:
            return TriggerResponse(success=False, message="Plan was empty")

    def display_plan(self):
        (success, self.plan, _, _) = self.move_group.plan(self.start_position)

        # Not needed since move_group.plan supposedly does this already
        # display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        # display_trajectory.trajectory_start = self.robot.get_current_state()
        # display_trajectory.trajectory.append(self.plan)
        # self.display_trajectory_publisher.publish(display_trajectory)

        return TriggerResponse(success=success, message="")

    def run(self):
        rospy.loginfo("Running position node")

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try: 
        position_node = PositionNode()
        position_node.run()
    except KeyboardInterrupt:
        pass