import cv2

class RecorderClass:

    def __init__(self):

        self.writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (640, 480))    
        self.write_f = 1

    def record_video(self, frame):

        if self.write_f == 1:

            self.writer.write(frame)

            display_image = frame

            cv2.imshow("Image window", display_image)

        if cv2.waitKey(1) & 0xFF == 27:
            print("VideoStoped")
            self.writer.release()
            cv2.destroyAllWindows()

            self.write_f = 0
