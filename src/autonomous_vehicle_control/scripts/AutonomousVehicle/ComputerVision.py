import cv2
import numpy as np

class ComputerVision:

    def __init__(self, crop_height_up_below):

        # Store crop dimensions
        self.crop_height_up_below = crop_height_up_below

        # Store the down-scale percentage
        self.scale_percent = 20 

    def get_image_shape(self, frame):

        # Store original frame shape
        self.original_frame_shape = frame.shape

        # Calculate target downscale dimensions
        downscaled_width = int(self.original_frame_shape[1] * self.scale_percent / 100)
        downscaled_height = int(self.original_frame_shape[0] * self.scale_percent / 100)
        
        # Store target downscale dimensions
        self.downscaled_dim = (downscaled_width, downscaled_height)   

    def crop_frame(self, frame):

        # Crop the frame and return 
        return frame[self.crop_height_up_below[0]:self.crop_height_up_below[1],:]

    def flatten_frame(self, frame):
        
        # Remove frame dimensions
        return frame.reshape(-1)

    def computer_vision_algorithm(self, frame):

        # Crop Frame
        cropped_frame = self.crop_frame(frame)

        # Convert to grayscale
        grayscale = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Downscale the frame
        resized = cv2.resize(grayscale, self.downscaled_dim, interpolation = cv2.INTER_AREA)

        # Apply threshold
        _,thresh1 = cv2.threshold(resized,200,1,cv2.THRESH_BINARY)

        # Return flattened frame
        return self.flatten_frame(thresh1)   

    def computer_vision_frame(self,frame):

        # Crop Frame
        cropped_frame = self.crop_frame(frame)
        
        # Convert to grayscale
        grayscale = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        # Downscale the frame
        resized = cv2.resize(grayscale, self.downscaled_dim, interpolation = cv2.INTER_AREA)

        # Apply threshold
        _,thresh1 = cv2.threshold(resized,200,100,cv2.THRESH_BINARY)

        # Return thresold frame
        return thresh1
