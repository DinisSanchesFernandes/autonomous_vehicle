#! /usr/bin/env python3

#/autonomous_vehicle_control/left_axle_controller/command
#/autonomous_vehicle_control/left_motor_controller/command
#/autonomous_vehicle_control/right_axle_controller/command
#/autonomous_vehicle_control/right_motor_controller/command


import time
import rospy
from std_msgs.msg import Float64
import numpy as np
import keyboard as key
from sensor_msgs.msg import Joy
from std_srvs.srv import Empty


def main_talker():

    autonomous_vehicle = AutonomousVehicle()

    while not rospy.is_shutdown():

        autonomous_vehicle.drive_autonomous_vehicle(autonomous_vehicle.vehicle_angle,autonomous_vehicle.vehicle_speed)
    
        autonomous_vehicle.rate.sleep()





class AutonomousVehicle:

    def __init__(self):

        # Define publisher and node to control joints in simulation
        self.left_axle_controller_pub = rospy.Publisher('/autonomous_vehicle/left_axle_controller/command', Float64, queue_size=10)
        self.right_axle_controller_pub = rospy.Publisher('/autonomous_vehicle/right_axle_controller/command', Float64, queue_size=10)
        self.left_motor_controller_pub = rospy.Publisher('/autonomous_vehicle/right_motor_controller/command', Float64, queue_size=10)
        self.right_motor_controller_pub = rospy.Publisher('/autonomous_vehicle/left_motor_controller/command', Float64, queue_size=10)
        rospy.init_node('autonomous_vehicle', anonymous=True)

        # Define publisher and node to input xbox controller
        rospy.Subscriber("/joy", Joy, self.callback_autonomous_vechicle)

        self.rate = rospy.Rate(10)

        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)


        self.vehicle_speed = 0
        self.vehicle_angle = 0
        self.button_A = 0
        self.button_B = 0

    def drive_autonomous_vehicle(self, X, Y):


        autonomous_vehicle_angle = (X * 0.52)
        autonomous_vehicle_speed = -(np.sqrt((X**2 + Y**2))* 8)
        print("Speed: ", autonomous_vehicle_speed,"Angle: ",autonomous_vehicle_angle)
        self.left_axle_controller_pub.publish(autonomous_vehicle_angle)
        self.right_axle_controller_pub.publish(autonomous_vehicle_angle)
        self.right_motor_controller_pub.publish(autonomous_vehicle_speed)
        self.left_motor_controller_pub.publish(autonomous_vehicle_speed)
        if self.button_A == 1:
            self.pause()
            time.sleep(4)
            print("Unpause")
            self.unpause() 
            
        if self.button_B == 1:
            print("Unpause")
            self.unpause() 


    def callback_autonomous_vechicle(self,data):

        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
        self.vehicle_speed= data.axes[1]
        self.vehicle_angle = data.axes[0]
        self.button_A = data.buttons[0]
        self.button_B = data.buttons[1]

if __name__ == '__main__':

    try:
        main_talker()
    except rospy.ROSInterruptException:
        pass