#! /usr/bin/env python
import rospy
import actionlib
import sys
from std_msgs.msg import String
import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
from time import sleep

args = int(sys.argv[1])

rospy.init_node('grip')
print("1")
        
pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)
command = outputMsg.Robotiq2FGripper_robot_output()
command.rACT = 1
command.rGTO = 1
command.rSP  = 255
command.rFR  = 150
print("2")

pub.publish(command)   
        
openGripper = args == 0
print("3")

command = outputMsg.Robotiq2FGripper_robot_output()
if not openGripper:
    command.rPR = 255
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    print("4")

else:
    command.rPR = 0
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 150
    print("5")

sleep(1)
print(command)
pub.publish(command)   

