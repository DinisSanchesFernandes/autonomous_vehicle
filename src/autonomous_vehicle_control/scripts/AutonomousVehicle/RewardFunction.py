import numpy as np

class RewardFunction:

    def __init__(self):

        # Store name of each track section
        self.track_position_array = ['Straight 1','Turn 1 (1)','Turn 1 (2)','Cross 1',
            'Straight 2','Cross 2','Turn 2 (1)','Turn 2 (2)']
        
        # Store parameters of each section ideal trajectory
        self.driving_line_array = [[1.775],[-0.675, 4.923, 2.45],[-0.675, 4.923, 2.45],
            [-0.675,0.723,1.75],[1.075],[-0.675, -0.677, 1.75],[-0.675, -4.877, 2.45],[-0.675, -4.877, 2.45]]

        # Store current track section
        self.current_track_section = 0 

        # Rewards
        self.penalty_section = -10
        self.penalty_track_limits = -10
        self.penalty_driving_line_error = -1
        self.threshold_track_limits = 0.25
        self.threshold_vehicle_speed = 3
        self.penalty_speed = -1
        self.reward_speed = 1

        self.reward_lap = 40
        self.reward_section = 20


    def driving_line_error_turn(self, xy, HKr):

        # Store the vehicle's coodinates
        X = xy[0]
        Y = xy[1]

        # Store the current section ideal trajectory parameters
        H = HKr[0]
        K = HKr[1]
        r = HKr[2]

        # Calculate and return displacement
        return np.abs(np.sqrt( (X - H)**2 + (Y - K)**2 ) - r )

    def driving_line_error_straight(self, xy, Xl):

        # Store the vehicle's X coodinate
        X = xy[0]

        # Store the current section ideal trajectory parameter
        Xl = Xl[0]

        # Calculate and return displacement
        return np.abs(Xl - X)

    def driving_line_error(self, xy):

        # Get vehicle's current coordinates
        vehicle_track_section = self.vehicle_track_position(xy)

        # Get current section idealtrajectory parameters
        driving_line_coord = self.driving_line_array[vehicle_track_section]

        # Destinguish striaght line vs turn
        if len(driving_line_coord) > 1:

            # Return displacement when in turn
            return self.driving_line_error_turn(xy, driving_line_coord), vehicle_track_section

        else:

            # Return displacement when in straight
            return self.driving_line_error_straight(xy, driving_line_coord), vehicle_track_section   

    
    def vehicle_track_position(self, xy):

        X = xy[0]
        Y = xy[1]

        # Turn 1 (1)
        if Y > 4.923:

            return 1

        # Turn 2 (2)
        elif Y < -4.877:

            return 7

        elif X < -0.675:

            # Turn 1 (2)
            if Y > 0:

                return 2

            # Turn 2 (1)
            else:

                return 6

        # Straight 1
        elif X > 1.425:

            return 0

        # Cross 2
        elif Y < -0.677:

            return 5
        
        # Cross 1
        elif Y > 0.723:

            return 3

        # Straight 2
        else:

            return 4

    def restart_reward(self, track_section):

        self.current_track_section = track_section


    # Return Reward, Done
    def get_reward_1(self, xy, wheel_speed):

        wheel_speed = np.clip(wheel_speed, 0, 6)

        error, track_section = self.driving_line_error(xy)

        if track_section != self.current_track_section:

            if track_section == self.current_track_section + 1:
                
                if track_section == 1 and self.current_track_section == 0:
                
                    self.current_track_section = track_section
                    #print("Reward track section")
                    return self.reward_lap, 0
                
                else:

                    self.current_track_section = track_section
                    #print("Reward track section")
                    return self.reward_section, 0


            elif self.current_track_section == 7 and track_section == 0:
                self.current_track_section = track_section
                return self.reward_section, 0
            
            else:
                #print("Penalty section")
                #print("Before: ", self.current_track_section)
                #print("Current: ",track_section)
                return self.penalty_section, 1

        elif error > self.threshold_track_limits:

            #print("Penalty Track Limits")
            return self.penalty_track_limits, 1

        else:
            
            if wheel_speed < 3:

                speed_val = self.penalty_speed

            else:

                speed_val = self.reward_speed

        
            return speed_val + error * self.penalty_driving_line_error, 0

    def get_reward_2(self, xy, wheel_speed):

        # The speed of the vehicle clipped
        wheel_speed = np.clip(wheel_speed, 0, 6)

        # Calulate the displacement of the vehicle and return the current section 
        error, track_section = self.driving_line_error(xy)

        # If the new track section is different from the current track section
        if track_section != self.current_track_section:

            # If the new track section is the correct  
            if track_section == self.current_track_section + 1:

                # Update the current track section with the new track section
                self.current_track_section = track_section

            # If the new track section is the last track section
            elif self.current_track_section == 7 and track_section == 0:
            
                # Update the current track section with the first track section
                self.current_track_section = track_section
            
            # If the vehicle jumped is not in the correct track section
            else:

                # Return the penalty for jumping a track section and end episode
                return self.penalty_section, 1

        # If the displacment of the vehicle is higher that the respective threshold
        if error > self.threshold_track_limits:

            # Return the penalty for exciding track limits and end episode
            return self.penalty_track_limits, 1

        # If the the displacement did not exceed track limits
        else:
            
            # If the speed of the vehicle is smaller than speed threshold
            if wheel_speed < self.threshold_vehicle_speed:

                # Get a penalty
                speed_val = self.penalty_speed

            # If the speed of the vehicle is higher than speed threshold
            else:

                # Get a reward
                speed_val = self.reward_speed

            # Return the Reward and continue episode
            return speed_val + error * self.penalty_driving_line_error, 0




