#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import TwistStamped, Twist
import numpy as np
import math
import pyquaternion
import tf
# Node that alters twist messages to restrict robot movement to a small workspace

class TwistFilter:
    def __init__(self):
        rospy.init_node('twist_filter', anonymous=False)
        rospy.Subscriber("/twist_cmd_raw", Twist, self.twist_callback)
        self.twist_pub = rospy.Publisher("/twist_controller/command", Twist, queue_size=10)
        self.twist_msg = TwistStamped()

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()


    def twist_callback(self, twist_msg: TwistStamped):
        # Taken from https://github.com/acosgun/deep_handover/blob/master/src/state_machine.py
        try:
            camera_t, camera_r = self.tf_listener.lookupTransform('base_link','tool0', rospy.Time())
            # rospy.loginfo(camera_t)
            safe_trans_v, safe_rot_v = get_safety_return_speeds(camera_t, camera_r)

            q_camera = pyquaternion.Quaternion(camera_r[3],camera_r[0],camera_r[1],camera_r[2])
            ctrl = twist_msg
            tran_v = q_camera.rotate((ctrl.linear.x,ctrl.linear.y,ctrl.linear.z))
            rot_v = q_camera.rotate((ctrl.angular.x,ctrl.angular.y,ctrl.angular.z))

            safe_trans_v, safe_rot_v = get_safety_return_speeds(camera_t,camera_r)

            msg = Twist()
            msg.linear.x, msg.linear.y, msg.linear.z = tran_v
            msg.angular.x, msg.angular.y, msg.angular.z = rot_v

            # msg.linear.x += safe_trans_v[0]
            # msg.linear.y += safe_trans_v[1]
            # msg.linear.z += safe_trans_v[2]

            # msg.angular.x += safe_rot_v[0]
            # msg.angular.y += safe_rot_v[1]
            # msg.angular.z += safe_rot_v[2]

            # Scale the linear and angular speeds
            v_speed = 0.10 #m/s
            r_speed = math.radians(20) #deg/s
            msg.linear.x *= v_speed
            msg.linear.y *= v_speed
            msg.linear.z *= v_speed
            msg.angular.x *= r_speed
            msg.angular.y *= r_speed
            msg.angular.z *= r_speed

            rospy.loginfo(msg)
            self.twist_pub.publish(msg)

        except (tf.LookupException,tf.ExtrapolationException) as e:
            rospy.logwarn(f"twist_filter: TF exception: {e}")
        

# Taken from https://github.com/acosgun/deep_handover/blob/master/src/state_machine.py
def get_safety_return_speeds(camera_t, camera_r):

    safe_trans_v = [0.0,0.0,0.0]
    safe_rot_v = np.array([0.0,0.0,0.0])

    q_camera = pyquaternion.Quaternion(camera_r[3],camera_r[0],camera_r[1],camera_r[2])
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

if __name__ == "__main__":
    node = TwistFilter()
    rospy.spin()
