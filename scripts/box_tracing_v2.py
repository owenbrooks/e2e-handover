#!/usr/bin/env python

####TODO: add limits to motion (new_position_safe function) -> Done


import sys
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import WrenchStamped

from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg
from enum import Enum
import math
import copy
from time import sleep

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import MoveGroupActionFeedback
from actionlib_msgs.msg import GoalStatusArray


# positions:
# Comments are relative to table
# [-0.308, 0.272, 0.382, 0.713, 0.699, 0.039, -0.035] -> Back right
# minus y of 0.417
# [-0.318, -0.145, 0.380, 0.689, 0.724, -0.009, 0.017] -> front right 
# minus x of 0.275
# [-0.593, -0.148, 0.384, 0.764, 0.644, -0.034, -0.016] -> front left

# Cardboard box home joint angles: [-1.4105542341815394, -1.834656063710348, 0.42505350708961487, -1.3924554030047815, 1.5589529275894165, 0.5954916477203369]

# Plastic box home joint angles: [-1.0356820265399378, -1.9370468298541468, 0.43775904178619385, -1.7621453444110315, 1.6236332654953003, 0.011440815404057503]



class MoveDirection(Enum):
    # Cardinal directions with north being out from table
    STARTUP = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

state_transition_table = {
    MoveDirection.NORTH: MoveDirection.EAST, 
    MoveDirection.EAST: MoveDirection.SOUTH, 
    MoveDirection.SOUTH: MoveDirection.WEST, 
    MoveDirection.WEST: MoveDirection.NORTH, 
}


state_move_direction = {
    MoveDirection.NORTH: {'coord':'y', 'direction':1, 'position': 0.4}, 
    MoveDirection.EAST: {'coord':'x', 'direction':1, 'position': -0.25}, 
    MoveDirection.SOUTH: {'coord':'y', 'direction':-1, 'position': -0.27}, 
    MoveDirection.WEST: {'coord':'x', 'direction':-1, 'position': -0.65}, 
}

TURN_THRESHOLD = 7
    

class BoxTracerNode():
    def __init__(self):
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        ## MoveIt setup. See move_group_python_interface_tutorial.py from moveit_tutorials
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("box_tracing")

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.status_sub = rospy.Subscriber('/scaled_pos_joint_traj_controller/follow_joint_trajectory/status', GoalStatusArray, self.status_callback)

        self.current_state = MoveDirection.STARTUP
        #Note order from /joint_states topic can be different from moveit order
        self.home_joint_angles = [ 0.4377,-1.9370,-1.035682, -1.762, 1.6236, 0.011440]

        self.initial_forces = {'x': 0, 'y': 0, 'z' : 0}
        self.force_reading = 0
        self.in_motion = 0

    def status_callback(self, status_msg):
        #print(status_msg.status_list[1])
        self.in_motion = status_msg.status_list

    def force_callback(self, wrench_msg):
        self.force_reading = wrench_msg.wrench.force
        
    
    def gripper_state_callback(self, gripper_input_msg):
        #self.obj_det_state = obj_msg_to_enum[gripper_input_msg.gOBJ]
        if self.current_state == MoveDirection.STARTUP:
            self.current_state = MoveDirection.NORTH

    def new_position_safe(self, new_pose):
        safe = True

        if new_pose.position.x > -0.2 or new_pose.position.x < -0.7:
            safe = False
        elif new_pose.position.y > 0.4 or new_pose.position.y < -0.27:
            safe = False

        return safe

    def move_to_position(self, state):
        plan_spacing = 0.1
        new_pose = self.move_group.get_current_pose().pose
        waypoints = []

        coord = state_move_direction[state]['coord']

        current_pos = getattr(new_pose.position, coord)
        goal_pos = state_move_direction[state]['position']
        num_steps = round(abs(current_pos - goal_pos) / plan_spacing)

        for i in range(int(num_steps)):
            setattr(new_pose.position, coord, getattr(new_pose.position, coord) + plan_spacing*state_move_direction[state]['direction'])
            waypoints.append(copy.deepcopy(new_pose))

        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

        run_flag = raw_input("Valid Trajectory? [y to run]:")

        if run_flag =="y":
            rospy.loginfo("Moving along box!")
            self.move_group.execute(plan, wait=False)
        else:
            rospy.signal_shutdown("Robot could not move to start location, ending execution!")
        

    def backoff(self, state):
        # Move back 5 cm
        backoff_distance = 0.1
        new_pose = self.move_group.get_current_pose().pose
        motion_direction = state_move_direction[state]

        # Negative as backoff is reversing
        new_pose.position.x -= motion_direction.x * backoff_distance
        new_pose.position.y -= motion_direction.y * backoff_distance


        self.move_group.set_pose_target(new_pose)
        #wait means wait for movement to to finish before executing any more code
        rospy.loginfo("Backing off!")
        plan_sucessful = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()


    
    def run(self):
        rospy.loginfo("Running force thresholding baseline node")
        rate = rospy.Rate(10)

        while self.current_state == MoveDirection.STARTUP and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rate.sleep()

        rospy.loginfo("Initializing gripper")
        # initialise the gripper
        grip_cmd = activate_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        # close the gripper to start
        grip_cmd = close_gripper_msg()
        self.gripper_pub.publish(grip_cmd)

        rospy.loginfo("Moving to start location")
        # move to start state
        plan = self.move_group.plan(self.home_joint_angles)

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

        run_flag = raw_input("Valid Trajectory? [y to run]:")

        if run_flag =="y":
            self.move_group.execute(plan, wait=True)
            self.current_state = MoveDirection.NORTH
        else:
            rospy.signal_shutdown("Robot could not move to start location, ending execution!")

        while self.force_reading == 0 and not rospy.is_shutdown():
            rate.sleep()

        # Initialise force sensor measurements
        self.initial_forces['x'] = self.force_reading.x
        self.initial_forces['y'] = self.force_reading.y
        self.initial_forces['z'] = self.force_reading.z
        rospy.loginfo("Initial force values: Fx = %.4f, Fy = %.4f, Fz = %.4f", self.initial_forces['x'], self.initial_forces['y'], self.initial_forces['z'])
        
        #breakpoint()
        
        while not rospy.is_shutdown():

            #Begin a motion in relevant direction
            rospy.loginfo("Current motion state: %s", self.current_state)
            self.move_to_position(self.current_state)
            rate.sleep()

            not_moving = False
            seen_one = False
            while not not_moving or not seen_one:
            # while not (not_moving and seen_one):
                force_mag = compute_force_magnitude(self.force_reading, self.initial_forces)
                rospy.loginfo("Force magnitude: %.4f", force_mag)
                
                not_moving = all([s.status == 3 for s in self.in_motion])
                if not seen_one:
                    seen_one = any([s.status == 1 for s in self.in_motion])

                if not_moving and seen_one:
                    print("Motion stopped")
                    print(self.in_motion)

                if force_mag > TURN_THRESHOLD:
                    rospy.loginfo("Force threshold exceeded, turning!")
                    execution_counter = 11
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    self.backoff(self.current_state)
                    self.current_state = state_transition_table[self.current_state]
                    sleep(1)
                    rate.sleep()
                else:
                    rate.sleep()
                    if not_moving and seen_one:
                        rospy.loginfo("End of execution")
                        self.move_group.stop()
                        self.move_group.clear_pose_targets()
                        self.current_state = state_transition_table[self.current_state]



def compute_force_magnitude(force_reading, initial_force):
    return math.sqrt((force_reading.x - initial_force['x'])**2 + (force_reading.y - initial_force['y'])**2 + (force_reading.z - initial_force['z'])**2)

if __name__ == "__main__":
    try: 
        box_tracer = BoxTracerNode()
        box_tracer.run()
    except KeyboardInterrupt:
        pass