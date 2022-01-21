from cv_bridge import CvBridge
import numpy as np
import rospy

use_tactile = True
try:
    from papillarray_ros_v2.msg import SensorState
    from e2e_handover import tactile
except ImportError:
    use_tactile = False
    print("Couldn't import papillarray")

class SensorManager():
    def __init__(self, use_camera_1, use_camera_2, use_force, use_tactile, use_segmentation):
        self.use_camera_1, self.use_camera_2, self.use_force = use_camera_1, use_camera_2, use_force
        self.use_tactile, self.use_segmentation = use_tactile, use_segmentation

        self.img_rgb_1 = None
        self.img_rgb_2 = None
        if use_camera_1:
            self.image_rgb_1_sub = rospy.Subscriber('/camera1/color/image_raw', Image, self.image_rgb_1_callback)
        if use_camera_2:
            self.image_rgb_2_sub = rospy.Subscriber('/camera2/color/image_raw', Image, self.image_rgb_2_callback)
        
        if use_force:
            self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
            self.raw_wrench_reading = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [fx, fy, fz, mx, my, mz]
            self.calib_wrench_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [fx, fy, fz, mx, my, mz]
            self.base_wrench_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # used to "calibrate" the force sensor since it gives different readings at different times
        
        self.abs_z_force = 0.0

        if use_tactile:
            self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.tactile_1_callback)
            self.tactile_2_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.tactile_2_callback)
        
            self.tactile_1_readings = [0.0]*(6+9*6) # 6 global readings plus 9 pillars with 6 readings each
            self.tactile_2_readings = [0.0]*(6+9*6) # 6 global readings plus 9 pillars with 6 readings each

        self.cv_bridge = CvBridge() # for converting ROS image messages to OpenCV images

        self.is_active = False

    def activate(self):
        self.is_active = True

    def sensors_ready(self):
        remaining_sensors = [
            self.use_camera_1 and self.img_rgb_1 is None,
            self.use_camera_2 and self.img_rgb_2 is None,
            self.use_force and self.raw_wrench_reading is None,
            self.use_tactile and (self.tactile_1_readings is None or self.tactile_2_readings is None),
        ]

        return any(remaining_sensors)

    def deactivate(self):
        self.is_active = False

        self.img_rgb_1 = None
        self.img_rgb_2 = None
        self.raw_wrench_reading = None
        self.tactile_1_readings = None
        self.tactile_2_readings = None

    def tactile_1_callback(self, sensor_msg):
        if self.is_active:
            value_list = tactile.sensor_state_to_list(sensor_msg)
            self.tactile_1_readings = value_list

    def tactile_2_callback(self, sensor_msg):
        if self.is_active:
            value_list = tactile.sensor_state_to_list(sensor_msg)
            self.tactile_2_readings = value_list

    def force_callback(self, wrench_msg):
        if self.is_active:
            self.abs_z_force = abs(self.base_wrench_array[2] - wrench_msg.wrench.force.z)

            self.raw_wrench_reading = np.array([wrench_msg.wrench.force.x, wrench_msg.wrench.force.y, wrench_msg.wrench.force.z, 
                wrench_msg.wrench.torque.x, wrench_msg.wrench.torque.y, wrench_msg.wrench.torque.z])

            # Subtracts a previous wrench reading to act as a kind of calibration
            self.calib_wrench_array = self.raw_wrench_reading - self.base_wrench_array
        else: 
            # Continuously calibrates the force sensor unless inference or recording is activated
            self.base_wrench_array = self.raw_wrench_reading

    def image_rgb_1_callback(self, image_msg):
        if self.is_active:
            try:
                self.img_rgb_1 = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                rospy.logerr(e)

    def image_rgb_2_callback(self, image_msg):
        if self.is_active:
            try:
                self.img_rgb_2 = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                rospy.logerr(e)