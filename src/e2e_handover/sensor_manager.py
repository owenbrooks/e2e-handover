from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
import numpy as np
import rospy

imported_tactile = True
try:
    from papillarray_ros_v2.msg import SensorState
    from papillarray_ros_v2.srv import BiasRequest
    from e2e_handover import tactile
except ImportError:
    imported_tactile = False
    print("Couldn't import papillarray")

class SensorManager():
    def __init__(self, sensor_params):
        (self.use_rgb_1, self.use_rgb_2, self.use_depth_1, self.use_depth_2, 
            self.use_force, self.use_tactile, self.use_segmentation, self.remove_background) = (
                sensor_params['use_rgb_1'], sensor_params['use_rgb_2'], 
                sensor_params['use_depth_1'], sensor_params['use_depth_2'],
                sensor_params['use_force'], sensor_params['use_tactile'] and imported_tactile, sensor_params['use_segmentation'],
                sensor_params['remove_background']
            )

        self.img_rgb_1 = None
        self.img_rgb_2 = None
        if self.use_rgb_1:
            self.image_rgb_1_sub = rospy.Subscriber('/camera1/color/image_raw', Image, self.image_rgb_1_callback)
        if self.use_rgb_2:
            self.image_rgb_2_sub = rospy.Subscriber('/camera2/color/image_raw', Image, self.image_rgb_2_callback)
        
        self.img_depth_1 = None
        self.img_depth_2 = None
        if self.use_depth_1:
            self.image_depth_1_sub = rospy.Subscriber('/camera1/aligned_depth_to_color/image_raw', Image, self.image_depth_1_callback)
        if self.use_depth_2:
            self.image_depth_2_sub = rospy.Subscriber('/camera2/aligned_depth_to_color/image_raw', Image, self.image_depth_2_callback)
        
        if self.use_force:
            self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
            self.raw_wrench_reading = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [fx, fy, fz, mx, my, mz]
            self.calib_wrench_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # [fx, fy, fz, mx, my, mz]
            self.base_wrench_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # used to "calibrate" the force sensor since it gives different readings at different times
        
        self.abs_z_force = 0.0

        if self.use_tactile:
            self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.tactile_1_callback)
            self.tactile_2_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.tactile_2_callback)
        
            self.tactile_1_readings = None # [0.0]*(6+9*6) # 6 global readings plus 9 pillars with 6 readings each
            self.tactile_2_readings = None # [0.0]*(6+9*6) # 6 global readings plus 9 pillars with 6 readings each

            self.contactile_bias_srv = rospy.ServiceProxy("/hub_0/send_bias_request", BiasRequest)

        self.cv_bridge = CvBridge() # for converting ROS image messages to OpenCV images

        self.is_active = False

    def activate(self):
        self.is_active = True

    def sensors_ready(self):
        waiting_on_sensors = [
            ('rgb_1', self.use_rgb_1 and self.img_rgb_1 is None),
            ('rgb_2', self.use_rgb_2 and self.img_rgb_2 is None),
            ('depth_1', self.use_depth_1 and self.img_depth_1 is None),
            ('depth_2', self.use_depth_2 and self.img_depth_2 is None),
            ('force', self.use_force and self.raw_wrench_reading is None),
            ('tactile', self.use_tactile and (self.tactile_1_readings is None or self.tactile_2_readings is None)),
        ]

        remaining_sensor_names = [sensor[0] for sensor in waiting_on_sensors if sensor[1]]
        all_ready = len(remaining_sensor_names) == 0
        if not all_ready:
            rospy.logwarn(f"Waiting for: {remaining_sensor_names}")

        return all_ready

    def deactivate(self):
        self.is_active = False

        # self.img_rgb_1 = None
        # self.img_rgb_2 = None
        # self.raw_wrench_reading = None
        # self.tactile_1_readings = None
        # self.tactile_2_readings = None

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

    def image_depth_1_callback(self, image_msg):
        if self.is_active:
            try:
                self.img_depth_1 = self.cv_bridge.imgmsg_to_cv2(image_msg)
            except CvBridgeError as e:
                rospy.logerr(e)

    def image_depth_2_callback(self, image_msg):
        if self.is_active:
            try:
                self.img_depth_2 = self.cv_bridge.imgmsg_to_cv2(image_msg)
            except CvBridgeError as e:
                rospy.logerr(e)
