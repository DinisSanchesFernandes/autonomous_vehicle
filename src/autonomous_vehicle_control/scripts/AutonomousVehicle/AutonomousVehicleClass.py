from telnetlib import NOP
import rospy
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import gazebo_msgs.msg
import numpy as np


class AutonomousVehicle:

    def __init__(self):

        # Init Node 
        rospy.init_node('autonomous_vehicle_control', anonymous=True)

        # Init node publishers to control angle of axle
        self.left_axle_controller_pub = rospy.Publisher(
            '/autonomous_vehicle/left_axle_controller/command', Float64, queue_size=10)
        self.right_axle_controller_pub = rospy.Publisher(
            '/autonomous_vehicle/right_axle_controller/command', Float64, queue_size=10)
        
        # Init node publishers to control back wheels speed 
        self.left_motor_controller_pub = rospy.Publisher(
            '/autonomous_vehicle/right_motor_controller/command', Float64, queue_size=10)
        self.right_motor_controller_pub = rospy.Publisher(
            '/autonomous_vehicle/left_motor_controller/command', Float64, queue_size=10)
        
        # Init node subscriber to get position of vehicle
        # This generates a callback to the self.callback_vehicle_state function
        rospy.Subscriber("/gazebo/model_states",
                         gazebo_msgs.msg.ModelStates, self.callback_vehicle_state)

        # Define variables for image processing
        self.cv_image_bridge = CvBridge()

        # Init node subscriber to get image frames from camera plug in
        # This generates a callback to the self.callback_camera_autonomous_vehicle function
        # The callback is done for every new frame published in the topic
        self.ros_image = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.callback_camera_autonomous_vehicle)

        # Init variables for camera frames 
        self.frame_cnt = 0

        # Init variables for vehicle position in simulation
        self.vehicle_pose_xy = [None, None]

    def drive_autonomous_vehicle(self, autonomous_vehicle_angle, autonomous_vehicle_speed):

        # Publish data to topics to control axle angle  
        self.left_axle_controller_pub.publish(autonomous_vehicle_angle)
        self.right_axle_controller_pub.publish(autonomous_vehicle_angle)

        # Publish data to topics to control back wheel speed
        self.right_motor_controller_pub.publish(-autonomous_vehicle_speed)
        self.left_motor_controller_pub.publish(-autonomous_vehicle_speed)

    def callback_camera_autonomous_vehicle(self, data):
      
        try:
            
            # Increment the frame counter 
            self.frame_cnt += 1

            # Convert the received frame  
            self.camera_frame = self.cv_image_bridge.imgmsg_to_cv2(
                data, "bgr8")

        # In case of error converting the frame
        except CvBridgeError as e:
            self.frame_cnt = 0
            print(e)

    def get_frame_reject_N(self, N_frames_to_reject):

        # Reset the frame counter
        self.frame_cnt = 0

        # Reject N-1 frames
        while self.frame_cnt < N_frames_to_reject:
            pass

        # Return N Frame
        return self.camera_frame

    def callback_vehicle_state(self, data):

        # Store the position of the vehicle to the variable
        self.vehicle_pose_xy = [
            data.pose[2].position.x, data.pose[2].position.y]

    def get_autonomous_vehicle_position(self):

        # Return position of the vehicle
        return self.vehicle_pose_xy


    def get_autonomous_vehicle_camera_frame(self):

        self.frame_cnt = 0
        return self.camera_frame

    def get_frames_N(self):

        return self.frame_cnt

    def init_vehicle_position(self):

        rospy.wait_for_service("/gazebo/model_states")

    def is_vehicle_frame_not_empty(self):

        return self.frame_cnt

    def is_vehicle_state_not_empty(self):

        if self.vehicle_pose_xy[0] == None:
            return False
        else:
            return True


