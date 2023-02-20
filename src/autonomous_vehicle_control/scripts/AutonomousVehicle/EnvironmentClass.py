
from cmath import pi
from AutonomousVehicle.AutonomousVehicleClass import AutonomousVehicle
from AutonomousVehicle.ComputerVision import ComputerVision
from AutonomousVehicle.RewardFunction import RewardFunction
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import rospy
import datetime
from std_srvs.srv import Empty
import cv2
import numpy as np

class Environment:

    def __init__(self):

        # Init Class
        self.autonomous_vehicle = AutonomousVehicle()
        self.computer_vision = ComputerVision((144, 320))
        self.reward_function = RewardFunction()

        # Store Starting position values
        self.starting_position = 0
        self.starting_position_array = [0, 2, 4, 3]
        self.starting_array_position_x = [1.81, -3.09, 1.11, 1.03]
        self.starting_array_position_y = [-4.39, 4.347, 0.7, 1.12]
        self.starting_array_position_z = [0.06, 0.062, 0.06, 0.06]
        self.starting_array_orientation_z = [-0.7,-0.78, -0.73, -0.72]
        self.starting_array_orientation_w = [0.7, -0.62, -0.69, -0.69]
        
        # Init variables to set position 
        self.set_position_msg = ModelState()
        self.set_position_msg.model_name = "autonomous_vehicle"
        self.set_position_msg.pose.position.x = self.starting_array_position_x[0]
        self.set_position_msg.pose.position.y = self.starting_array_position_y[0] 
        self.set_position_msg.pose.position.z = self.starting_array_position_z[0]
        self.set_position_msg.pose.orientation.x = 0
        self.set_position_msg.pose.orientation.y = 0
        self.set_position_msg.pose.orientation.z = self.starting_array_orientation_z[0]
        self.set_position_msg.pose.orientation.w = self.starting_array_orientation_w[0]

        # Init reset service
        self.reset_simulation_str = '/gazebo/reset_simulation'
        try:
            rospy.wait_for_service(self.reset_simulation_str)
            self.reset_simulation = rospy.ServiceProxy(self.reset_simulation_str, Empty)

        except(rospy.ServiceException, rospy.ROSException):
            print("Error: Reset Sim Service Wait Failed")
            return False    

        # Init pause unpause service
        self.pause_str = '/gazebo/pause_physics'
        self.unpause_str = '/gazebo/unpause_physics'
        try:
            rospy.wait_for_service(self.unpause_str)
            rospy.wait_for_service(self.pause_str)
            self.pause = rospy.ServiceProxy(self.pause_str, Empty)
            self.unpause = rospy.ServiceProxy(self.unpause_str, Empty)
        except(rospy.ServiceException, rospy.ROSException):
            print("Error: Unpause Pause Service Wait Failed")
            return False    
       
        # Init set model state service
        self.set_position_str = '/gazebo/set_model_state'
        try:
            rospy.wait_for_service(self.set_position_str)
            self.set_position = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        except(rospy.ServiceException, rospy.ROSException):
            print("Error: Set Model State Service Wait Failed")
            return False   

        # Vision
        self.frames_to_reject = 2

        self.action_space_dimension = 2  

        self.video_render_count = 0

        # Rate
        self.rate = rospy.Rate(10)

    def set_position_func(self):
        
        self.set_position_msg.pose.position.x = self.starting_array_position_x[self.starting_position]
        self.set_position_msg.pose.position.y = self.starting_array_position_y[self.starting_position] 
        self.set_position_msg.pose.position.z = self.starting_array_position_z[self.starting_position]
        self.set_position_msg.pose.orientation.z = self.starting_array_orientation_z[self.starting_position]
        self.set_position_msg.pose.orientation.w = self.starting_array_orientation_w[self.starting_position]
            
        self.reward_function.restart_reward(self.starting_position_array[self.starting_position])
        self.set_position(self.set_position_msg)
        self.starting_position +=1
        if self.starting_position > len(self.starting_position_array) - 1:
            self.starting_position = 0


        

    def wait_for_init(self):

        
        #print("Waiting for Camera")
        while not self.autonomous_vehicle.is_vehicle_frame_not_empty():
            self.rate.sleep()
        #print("Success")
        #print("Waiting for Position")
        while not self.autonomous_vehicle.is_vehicle_state_not_empty():
            self.rate.sleep()
        #print("Success")

    def render_init(self):

        self.video_render_count += 1
        now = datetime.datetime.now()

        writer= cv2.VideoWriter("/home/dinis/ROS/autonomous_vehicle/src/autonomous_vehicle_control/scripts/RenderedVideos/Eval_"+str(now)+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))    

        return writer 

    def render(self,writer):

        frame = self.autonomous_vehicle.get_autonomous_vehicle_camera_frame()

        writer.write(frame)

    def render_close(self, writer):

        writer.release()

        
    def close(self):
        rospy.wait_for_service(self.pause_str)
        self.pause()

    def get_state_reward(self,wheel_speed):

        # Get frame after rejecting N frames
        frame = self.autonomous_vehicle.get_frame_reject_N(self.frames_to_reject)
        
        # Get position of the vehicle
        position = self.autonomous_vehicle.get_autonomous_vehicle_position()
        
        # Get reward and done
        reward, done = self.reward_function.get_reward_2(position, wheel_speed)
        
        # Process the frame to get the state
        flatten = self.computer_vision.computer_vision_algorithm(frame)

        # Return state reward and done
        return flatten, reward, done

    def step(self, action):

        # Adjust the action of the neural nerwork 
        angle = (action[0] - 0.5) * 0.54
        
        # Adjust the action of the neural nerwork
        speed = action[1] * 6

        # Clip angle value
        angle = np.clip(angle, -0.52, 0.52)
        
        # Clip speed value
        speed = np.clip(speed, 0, 6)

        # Execute the action 
        self.autonomous_vehicle.drive_autonomous_vehicle(angle, speed)

        # Get the state, reward and done 
        state, reward, done = self.get_state_reward(speed)

        # Return State reward and done
        return state, reward, done

    def reset(self):

        #self.reward_function.restart_reward()

        rospy.wait_for_service(self.reset_simulation_str)
        self.reset_simulation()

        rospy.wait_for_service(self.set_position_str)
        self.set_position_func()

        rospy.wait_for_service(self.unpause_str)
        self.unpause()

        self.wait_for_init()

        frame = self.autonomous_vehicle.get_autonomous_vehicle_camera_frame()

        return self.computer_vision.computer_vision_algorithm(frame)

    def get_action_space_dimensions(self):

        return self.action_space_dimension

    def get_state_space_dimensions(self):

        state = self.reset()

        return state.shape[0]

    def environment_init(self):

        rospy.wait_for_service(self.reset_simulation_str)
        self.reset_simulation()
        #print("Reset Initi")
        self.wait_for_init()
        #print("Reset close")
        frame = self.autonomous_vehicle.get_autonomous_vehicle_camera_frame()

        return self.computer_vision.get_image_shape(frame)

