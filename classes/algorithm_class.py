import numpy as np
import cv2








class control_algorithm:
    def __init__(self):
        self.node = 0

    def run(self, robot_list, frame):
        
 
        pts = np.array(robot_list[-1].trajectory, np.int32)
        cv2.polylines(frame, [pts], False, (0, 0, 255), 4)
        print(len(robot_list[-1].trajectory))

        #logic for arrival condition
        if self.node == len(robot_list[-1].trajectory):
            #weve arrived
           
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = 0
            gamma = np.pi/2   #disregard
            freq = 0    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard


        #closed loop algorithm 
        else:
            #define target coordinate
            targetx = robot_list[-1].trajectory[self.node][0]
            targety = robot_list[-1].trajectory[self.node][1]

            #define robots current position
            robotx = robot_list[-1].position_list[-1][0]
            roboty = robot_list[-1].position_list[-1][1]
            
            #calculate error between node and robot
            direction_vec = [targetx - robotx, targety - roboty]
            error = np.sqrt(direction_vec[0] ** 2 + direction_vec[1] ** 2)
            if error < 40:
                self.node += 1
            
            cv2.arrowedLine(
                    frame,
                    (int(robotx), int(roboty)),
                    (int(targetx), int(targety)),
                    [100, 100, 100],
                    3,
                )
                
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = np.arctan2(-direction_vec[1], direction_vec[0])  - np.pi/2
            gamma = np.pi/2   #disregard
            freq = 5    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard
        
        
        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq