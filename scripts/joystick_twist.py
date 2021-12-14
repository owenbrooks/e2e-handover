#!/usr/bin/env python
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import TwistStamped
import numpy as np
import math
import pyquaternion

class JoystickTwist:

    def __init__(self):
        rospy.init_node('joystick_twist', anonymous=False)
        rospy.Subscriber("joy", sensor_msgs.msg.Joy, self.joystick_callback)
        self.twist_pub = rospy.Publisher("twist_cmd", TwistStamped, queue_size=10)
        self.twist_msg = TwistStamped

    def spin(self):
        # Publish the latest control message at 30Hz
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.ctrl_pub.publish(self.ctrl_msg)
            self.record_pub.publish(self.record_msg)
            r.sleep()

    def joystick_callback(self,data):
        # Get all the axes and button values
        left_x, left_y, trig_l, right_x, right_y, trig_r, dpad_x, dpad_y = data.axes
        btn_a, btn_b, btn_x, btn_y, bump_l, bump_r, back, menu, _, stick_l, stick_r, _, _ = data.buttons

        # Create a new robot control message
        twist_msg = TwistStamped()
        # Translation
        twist_msg.twist.linear.x = -deadband(left_x)
        twist_msg.twist.linear.y = - deadband(left_y)
        twist_msg.twist.linear.z = deadband(trig_l/2.0 - trig_r/2.0)
        # Rotation
        twist_msg.twist.angular.x =  deadband(right_y)
        twist_msg.twist.angular.y = - deadband(right_x)
        twist_msg.twist.angular.z = bump_r - bump_l

        self.twist_msg = twist_msg

def get_safety_return_speeds(camera_t, camera_r):

    safe_trans_v = [0.0,0.0,0.0]
    safe_rot_v = np.array([0.0,0.0,0.0])

    # #calculate safe quaternion based on camera position
    # angle_to_base = math.atan2(camera_t[1],camera_t[0]) + math.pi/2
    # safe_q = pyquaternion.Quaternion(axis=(0.0, 0.0, 1.0), radians=angle_to_base)
    # safe_q *= pyquaternion.Quaternion(axis=(0.0, 1.0, 0.0), degrees=180.0)
    #
    # angle_limit = 45 #degrees
    q_camera = pyquaternion.Quaternion(camera_r[3],camera_r[0],camera_r[1],camera_r[2])

    #calculate quaternion difference between camera and safe quaternion
    # difference_q = safe_q/q_camera
    # align_speed = min(2.0,max(0.0,(abs(difference_q.degrees)-angle_limit)/5.0))
    # align_speed *= np.sign(difference_q.degrees)
    # safe_rot_v = [difference_q.axis[0]*align_speed, difference_q.axis[1]*align_speed, difference_q.axis[2]*align_speed]

    cam_z = q_camera.rotate(np.array([0.0,0.0,1.0]))
    cam_y = q_camera.rotate(np.array([0.0,1.0,0.0]))

    #PAN TILT LIMITS
    full_rot_speed = 1.2 #degrees per second
    full_rot_speed_dist = math.radians(5) #degrees per second

    #get the normal to the plane that runs throught the robot z axis and the camera position
    v_plane_norm = np.array([-camera_t[1],camera_t[0],0.0])
    v_plane_norm /= np.linalg.norm(v_plane_norm)
    #project camera z axis onto plane
    z_proj = cam_z - np.dot(cam_z,v_plane_norm)*v_plane_norm
    #normalise z projection
    z_proj /= np.linalg.norm(z_proj)

    #PAN
    pan_angle_limit = math.radians(20) # degrees
    #get the pan rotation axis between camera z and plane
    pan_axis = -np.cross(z_proj,cam_z)
    #get the pan angle
    pan_angle = np.linalg.norm(pan_axis) #TODO This not correct way to get angle from cross product. It needs a sin
    #normalise pan rotation axis
    pan_axis /= np.linalg.norm(pan_axis)

    pan_return_speed = min(full_rot_speed,max(0,(pan_angle-pan_angle_limit)/full_rot_speed_dist*full_rot_speed))

    safe_rot_v += pan_axis * pan_return_speed

    #TILT LIMIT
    out_norm = np.array([camera_t[0],camera_t[1],0.0])
    out_norm /= np.linalg.norm(out_norm)
    tilt_axis = np.cross(z_proj,out_norm)
    tilt_sign_y = np.sign(np.dot(tilt_axis,v_plane_norm))
    tilt_sign_x = np.sign(np.dot(z_proj,out_norm))

    tilt_angle = math.asin(np.linalg.norm(tilt_axis))
    if tilt_sign_x < 0:
        tilt_angle = math.pi - tilt_angle
    tilt_angle *= tilt_sign_y

    tilt_angle_max = math.radians(30)
    tilt_angle_min = math.radians(-30)

    tilt_return_speed = -min(full_rot_speed,max(0,(tilt_angle-tilt_angle_max)/full_rot_speed_dist*full_rot_speed))
    tilt_return_speed += min(full_rot_speed,max(0,(tilt_angle_min-tilt_angle)/full_rot_speed_dist*full_rot_speed))

    # tilt_return_speed
    # print("%f %f" % (math.degrees(tilt_sign_x),math.degrees(tilt_angle)))

    safe_rot_v -= v_plane_norm * tilt_return_speed

    #LOOK UP
    if tilt_angle > -math.pi/4 and tilt_angle < math.pi/4:
        up_dir = np.array([0.0,0.0,1.0])
    else:
        up_dir = out_norm
    #project up direction onto camera xy plane
    up_proj = up_dir - np.dot(up_dir,cam_z)*cam_z
    #normalise project up direction
    up_proj /= np.linalg.norm(up_proj)
    #If up direction is withing +- 90deg  cam_y
    if np.dot(up_proj,-cam_y) > 0:
        #Get rotation speed from cross product
        safe_rot_v += -np.cross(up_proj,-cam_y)*5

    full_speed_dist = 0.05
    full_speed = 1.5
    #Floor
    safe_z = 0.4#0.15
    safe_trans_v[2] += max(0,(safe_z -camera_t[2])/full_speed_dist*full_speed)

    #Inner Cylinder
    cylinder_radius = 0.5# 0.35
    dist = math.sqrt( camera_t[0]**2 + camera_t[1]**2)
    cylinder_return_speed = max(0,(cylinder_radius-dist)/full_speed_dist*full_speed)
    safe_trans_v[0] += camera_t[0]/dist * cylinder_return_speed
    safe_trans_v[1] += camera_t[1]/dist * cylinder_return_speed

    #Outer Sphere
    sphere_radius = 0.9#0.7
    dist = math.sqrt( camera_t[0]**2 + camera_t[1]**2 + camera_t[2]**2)
    cylinder_return_speed = min(0,(sphere_radius-dist)/full_speed_dist*full_speed)
    safe_trans_v[0] += camera_t[0]/dist * cylinder_return_speed
    safe_trans_v[1] += camera_t[1]/dist * cylinder_return_speed
    safe_trans_v[2] += camera_t[2]/dist * cylinder_return_speed

    #back Wall
    wall_unit_norm = [0.7071,-0.7071]
    dist = camera_t[0] * wall_unit_norm[0] + camera_t[1] * wall_unit_norm[1]
    wall_return_speed = -min(0,dist/full_speed_dist*full_speed)
    safe_trans_v[0] += wall_unit_norm[0] * wall_return_speed
    safe_trans_v[1] += wall_unit_norm[1] * wall_return_speed

    return (safe_trans_v, safe_rot_v.tolist())

def deadband(var,band=0.2):
    var = max(-1.0,min(var,1.0))

    if var > band:
        return (var-band) / (1.0-band)

    if var < -band:
        return (var+band) / (1.0-band)
    return 0.0

if __name__ == "__main__":
    node = JoystickTwist()
    node.spin()