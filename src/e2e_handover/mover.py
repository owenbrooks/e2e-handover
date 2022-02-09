# Performs canned handover motion
import moveit_commander
import numpy as np
import os
import pickle
import rospkg
from sensor_msgs.msg import JointState
import sys
import rospy
from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState

# Plan method from moveit, modified so that it can accept a list of joint angles without triggering a bug
# http://docs.ros.org/en/noetic/api/moveit_commander/html/move__group_8py_source.html#l00613 
# replaces move_group.plan() method
def plan(self, joints=None):
    """Return a tuple of the motion planning results such as
    (success flag : boolean, trajectory message : RobotTrajectory,
    planning time : float, error code : MoveitErrorCodes)"""
    
    self.set_joint_value_target(joints)

    (error_code_msg, trajectory_msg, planning_time) = self._g.plan()

    error_code = MoveItErrorCodes()
    error_code.deserialize(error_code_msg)
    plan = RobotTrajectory()
    return (
        error_code.val == MoveItErrorCodes.SUCCESS,
        plan.deserialize(trajectory_msg),
        planning_time,
        error_code,
    )
  

class Mover():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.group_name = "manipulator" # default for the UR5
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # base, shoulder, elbow, wrist1, wrist2, wrist3
        self.retracted_goal_joint_position = list(np.deg2rad([90., -50., 100., -230., -90., 0.])) # only used if there is no saved plan
        # self.retracted_goal_joint_position = list(np.deg2rad([0., -90., 0., -90., 0., 0.])) # only used if there is no saved plan
        self.extended_goal_joint_position = list(np.deg2rad([-80., -145., -70., 33., 85., 0.0])) # only used if there is no saved plan
        
        self.setup_plan = None
        self.reach_plan = None
        self.return_plan = None

        rospack = rospkg.RosPack()
        package_dir = rospack.get_path('e2e_handover')
        self.setup_plan_path = os.path.join(package_dir, 'saved_trajectories', 'setup_plan.p')
        self.reach_plan_path = os.path.join(package_dir, 'saved_trajectories', 'reach_plan.p')
        self.return_plan_path = os.path.join(package_dir, 'saved_trajectories', 'return_plan.p')

        load_success = self.load_plans()
        if not load_success:
            create_success = self.create_plans()
            if not create_success:
                print("Unable to execute without a plan. Exiting.")
                sys.exit()

        self.setup()

    def load_plans(self) -> bool:
        try:
            with open(self.setup_plan_path, 'rb') as setup_plan_file:
                self.setup_plan = pickle.load(setup_plan_file)
            with open(self.reach_plan_path, 'rb') as reach_plan_file:
                self.reach_plan = pickle.load(reach_plan_file)
            with open(self.return_plan_path, 'rb') as return_plan_file:
                self.return_plan = pickle.load(return_plan_file)
        except FileNotFoundError:
            return False
        return True

    def create_plans(self) -> bool:
        print("Unable to load plans. Would you like to create plans? (y/n)")
        will_create = said_yes()
        if not will_create:
            return False

        # Make sure robot is in home position to start
        robot_homed = False
        while not robot_homed:
            print("Is the robot in the home position? (y/n)")
            robot_homed = said_yes()
            if not robot_homed:
                print("Please move the robot to the home position")

        # home->retracted
        print(self.retracted_goal_joint_position)
        success, setup_plan, _, _ = plan(self.move_group, self.retracted_goal_joint_position)
        if not success:
            return False

        print("Displaying setup plan. Execute? (y/n)")
        should_execute = said_yes()
        if should_execute:
            self.move_group.execute(setup_plan, wait=True)
            with open(self.setup_plan_path, 'wb') as setup_plan_file:
                pickle.dump(setup_plan, setup_plan_file)
                print(f'Dumped yaml to {setup_plan_file}')
        # retracted->extended
        success, reach_plan, _, _ = plan(self.move_group, self.extended_goal_joint_position)
        if not success:
            return False
        print("Displaying reach plan. Execute? (y/n)")
        should_execute = said_yes()
        if should_execute:
            self.move_group.execute(reach_plan, wait=True)        
            with open(self.reach_plan_path, 'wb') as reach_plan_file:
                pickle.dump(reach_plan, reach_plan_file)
        # extended->retracted
        success, return_plan, _, _ = plan(self.move_group, self.retracted_goal_joint_position)
        if not success:
            return False
        print("Displaying return plan. Execute? (y/n)")
        should_execute = said_yes()
        if should_execute:
            self.move_group.execute(return_plan, wait=True)        
            with open(self.return_plan_path, 'wb') as return_plan_file:
                pickle.dump(return_plan, return_plan_file)

        print("Saved plans.")

        return True
            
    def setup(self):
        rospy.loginfo("Setting up")
        self.move_group.execute(self.setup_plan, wait=True)

    def reach(self):
        rospy.loginfo("Reaching")
        self.move_group.execute(self.reach_plan, wait=True)

    def retract(self):
        rospy.loginfo("Retracting")
        self.move_group.execute(self.return_plan, wait=True) 

def said_yes() -> bool:
    response = input().lower().strip()
    if len(response) > 0:
        if response[0] == 'y':
            return True
        elif response[0] == 'n':
            return False
    return False