from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
from enum import Enum

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

def open_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rPR = 0
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command

def close_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rPR = 255
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command

def reset_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 0

    return command

def activate_gripper_msg():
    command = outputMsg.Robotiq2FGripper_robot_output()
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150

    return command