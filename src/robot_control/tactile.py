from papillarray_ros_v2.msg import SensorState
from papillarray_ros_v2.srv import BiasRequest
import rospy

global_force_torques = ['tact_gfX', 'tact_gfY', 'tact_gfZ', 'tact_gtX', 'tact_gtY', 'tact_gtZ']
pillar_disp_force = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ'] 
pillar_keys = []
for i in range(9):
    for key in pillar_disp_force:
        pillar_keys.append(f'tact_{i}_{key}')
papillarray_keys = global_force_torques + pillar_keys

def sensor_state_to_list(msg):
    values = [msg.gfX, msg.gfY, msg.gfZ, msg.gtX, msg.gtY, msg.gtZ]
    
    for pil in msg.pillars: # will be 9 pillars
        values += [pil.dX, pil.dY, pil.dZ, pil.fX, pil.fY, pil.fZ]
    
    return values

def request_bias():
    rospy.logwarn("requesting bias")
    rospy.wait_for_service('/hub_0/send_bias_request')
    try:
        bias_server = rospy.ServiceProxy('/hub_0/send_bias_request', BiasRequest)
        response = bias_server()
        if not response.result:
            rospy.logerr("Tactile sensor biasing unsuccessful.")
    except rospy.ServiceException as e:
        rospy.logerr("Tactile sensor biasing service failed: %s"%e)

    rospy.logwarn("Finished bias")