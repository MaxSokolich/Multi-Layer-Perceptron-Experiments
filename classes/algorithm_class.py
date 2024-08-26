import numpy as np
import cv2
import torch

from classes.MLP_controller import MLP_controller


class control_algorithm:
    def __init__(self, p1,p2,t1,t2):
        self.node = 0 
        self.counter = 0


        self.my_mlp_controller = MLP_controller()
        self.my_mlp_controller.N = 1

        start_robot1_x = float(p1[0])
        start_robot1_y = float(p1[1])

        start_robot2_x = float(p2[0])
        start_robot2_y = float(p2[1])
        
        target_robot1_x = float(t1[0])
        target_robot1_y = float(t1[1])

        target_robot2_x = float(t2[0])
        target_robot2_y = float(t2[1])

        print("robot1pos = {}, robot2pos = {}, robot1target= {}, robot2target = {}".format(p1,p2,t1,t2))


        Dx1 = np.array([target_robot1_x - start_robot1_x     ,     target_robot1_y - start_robot1_y])
        Dx2 = np.array([target_robot2_x - start_robot2_x     ,     target_robot2_y - start_robot2_y])

        #Dx = np.array([Dx1, Dx2])
        Dx = torch.tensor([Dx1, Dx2])


        self.freqs, self.alphas = self.my_mlp_controller.getControl(Dx)
        


    def run(self, robot_list, frame):
        #this gets called at each frame

        self.counter +=1

        if self.counter <= len(self.freqs):
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = self.alphas[self.counter]
            gamma = np.pi/2   #disregard
            freq = self.freqs[self.counter]    #CHANGE THIS EACH FRAME
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard

        else:
            Bx = 0 #disregard
            By = 0 #disregard
            Bz = 0 #disregard
            alpha = 0
            gamma = np.pi/2   #disregard
            freq = 0
            psi = np.pi/2      #disregard
            gradient = 0 # #disregard
            acoustic_freq = 0  #disregard
    
        
        return frame, Bx, By, Bz, alpha, gamma, freq, psi, gradient, acoustic_freq