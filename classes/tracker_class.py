from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread,QTimer
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage 
import time


from classes.fps_class import FPSCounter
    
#add unique crop length 
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)
    cropped_frame_signal = pyqtSignal(np.ndarray,np.ndarray)
  


    def __init__(self, parent):
        super().__init__(parent=parent)
        self.parent = parent
        self.cap = self.parent.cap 
        video = self.parent.videopath 
        #initiate control class
       
        
    
        self.fps = FPSCounter()
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videofps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self._run_flag = True
        self._play_flag = True
        self.mask_flag = False
    
        self.croppedmask_flag = True
        self.framenum = 0

        self.orientstatus = False
        self.autoacousticstatus = False
        
        #robot mask attributes
        self.robot_mask_lower = 0
        self.robot_mask_upper = 128
        self.robot_mask_dilation = 0  
        self.robot_mask_blur = 0
        self.robot_crop_length = 40
        self.robot_mask_flag = True
        self.robot_list = []

        #cell mask attributes
        self.cell_mask_lower = 0
        self.cell_mask_upper = 128
        self.cell_mask_dilation = 0
        self.cell_mask_blur = 0
        self.cell_crop_length = 40
        self.cell_mask_flag = False
        self.cell_list = []

        self.robot_mask = None
        self.maskinvert = False
        self.crop_length_record = 200
        
        self.exposure = 5000
        self.objective = 10


        self.arrivalthresh = 100
        self.RRTtreesize = 25
        self.memory = 15  #this isnt used as of now
     

        if video != 0:
            self.totalnumframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.totalnumframes = 0
           
        self.pix2metric =  0.28985 * self.objective #.29853 * self.objective#0.28985 * self.objective  
        
        

 
    
    def find_robot_mask(self,frame):
        """
        finds a mask of a given image based on a threshold value in black and white for ROBOTS
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.robot_mask_blur > 0:
            frame = cv2.blur(frame, (self.robot_mask_blur,self.robot_mask_blur))

        #threshold the mask
        #_, robot_mask = cv2.threshold(frame, robot_mask_thresh, 255, cv2.THRESH_BINARY)
        #robot_mask = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.robot_mask_blocksize, self.robot_mask_thresh)
        robot_mask = cv2.inRange(frame, self.robot_mask_lower, self.robot_mask_upper)
        
        if self.maskinvert:
            robot_mask = cv2.bitwise_not(robot_mask)

        #subtract cell from robot mask
        try:
            for cell in self.cell_list:    
                x,y,w,h = cell.cropped_frame[-1]
                blank = np.zeros((w, h), dtype=np.uint8) 
                robot_mask[y:y+w , x:x+h] = blank 
        except Exception:
            pass
     

        robot_mask = cv2.dilate(robot_mask, None, iterations=self.robot_mask_dilation)

        return robot_mask
    
       


    def track_robot(self, frame, robotmask):
        """
        Returns:
            cropped_mask: to visualize tracking parameters
        """
        if len(self.robot_list) > 0:
            for i in range(len(self.robot_list)): #for each bot with a position botx, boty, find the cropped frame around the bot
                try:
                    bot = self.robot_list[i]
                    #current cropped frame dim
                    x1, y1, w, h = bot.cropped_frame[-1]
                    x1 = max(min(x1, self.width), 0)
                    y1 = max(min(y1, self.height), 0)

                    #crop the frame and mask
                    croppedframe = frame[y1 : y1 + h, x1 : x1 + w]
                    croppedmask  = robotmask[y1 : y1 + h, x1 : x1 + w]
                
                
                    #label the mask
                    label_im, nb_labels = ndimage.label(croppedmask) 
                    sizes = ndimage.sum(croppedmask, label_im, range(nb_labels + 1)) 
                    num_bots=np.sum(sizes>50)
                    
                    if num_bots>0:
                        #find contours from the mask
                        contours, _ = cv2.findContours(croppedmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        max_cnt = contours[0]
                        for contour in contours:
                            if cv2.contourArea(contour) > cv2.contourArea(max_cnt): 
                                max_cnt = contour
                        area = cv2.contourArea(max_cnt)* (1/self.pix2metric**2)
                        
                        #find the center of mass from the mask
                        szsorted=np.argsort(sizes)
                        [ycord,xcord]=ndimage.center_of_mass(croppedmask,labels=label_im,index = szsorted[-(1)])
                        ndimage.binary_dilation
                        
                        #derive the global current location
                        current_pos = [xcord + x1,   ycord + y1] #xcord ycord are relative to the cropped frame. need to convert to the overall frame dim

                        #generate new cropped frame based on the new robots position
                        x1_new = int(current_pos[0] - bot.crop_length/2)
                        y1_new = int(current_pos[1] - bot.crop_length/2)
                        w_new = int(bot.crop_length)
                        h_new = int(bot.crop_length)
                        new_crop = [int(x1_new), int(y1_new), int(w_new), int(h_new)]


                        #find velocity:
                        if len(bot.position_list) > self.memory:
                            vx = (current_pos[0] - bot.position_list[-self.memory][0]) * (self.fps.get_fps()/self.memory) / self.pix2metric
                            vy = (current_pos[1] - bot.position_list[-self.memory][1]) * (self.fps.get_fps()/self.memory) / self.pix2metric
                            magnitude = np.sqrt(vx**2 + vy**2)

                            velocity = [vx,vy,magnitude]

                        else:
                            velocity = [0,0,0]

                
                        #find blur of original crop
                        blur = cv2.Laplacian(croppedframe, cv2.CV_64F).var()
                        
                        #store the data in the instance of RobotClasss
                        bot.add_frame(self.framenum)
                        bot.add_time(1/self.fps.get_fps()) #original in ms
                        bot.add_position([current_pos[0], current_pos[1]])
                        bot.add_velocity(velocity)
                        bot.add_crop(new_crop)
                        bot.add_area(area)
                        bot.add_blur(blur)
                        bot.set_avg_area(np.mean(bot.area_list))


                        #stuck condition
                        if len(bot.position_list) > self.memory and velocity[2] < 20 and self.parent.freq > 0:
                            stuck_status = 1
                        else:
                            stuck_status = 0
                        bot.add_stuck_status(stuck_status)

                        #this will toggle between the cropped frame display being the masked version and the original
                        if self.croppedmask_flag == False:
                            croppedmask = croppedframe

                    else:
                        if len(self.robot_list) > 0:
                            del self.robot_list[i]
                
                except Exception:
                    pass
                       
        
            #also crop a second frame at a fixed wdith and heihgt for recording the most recent robots suroundings
            if len(self.robot_list)>0:
                x1_record = int(bot.position_list[-1][0] - self.crop_length_record/2)
                y1_record = int(bot.position_list[-1][1] - self.crop_length_record/2)
                recorded_cropped_frame = frame[y1_record : y1_record + self.crop_length_record, x1_record : x1_record + self.crop_length_record]
                
                #adjust most recent bot crop_length 
                bot.crop_length = self.robot_crop_length
            else:
                recorded_cropped_frame = np.zeros((self.crop_length_record, self.crop_length_record, 3), dtype=np.uint8) 

        else:
            recorded_cropped_frame = np.zeros((self.crop_length_record, self.crop_length_record, 3), dtype=np.uint8) 
            croppedmask = np.zeros((310, 310, 3), dtype=np.uint8)
        
        return croppedmask, recorded_cropped_frame
    


    
    





    def display_hud(self, frame):
        
        display_frame = frame.copy()
        if len(self.robot_list) > 0:
            color = plt.cm.rainbow(np.linspace(1, 0.2, len(self.robot_list))) * 255
    
            for (botnum,botcolor) in zip(range(len(self.robot_list)), color):
               
                bot  = self.robot_list[botnum]
                x1, y1, w, h = bot.cropped_frame[-1]

                cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), botcolor, 4)
                cv2.putText(display_frame,str(botnum+1),(x1 + w,y1 + h),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4,color = (255, 255, 255))
                
                pts = np.array(bot.position_list, np.int32)
                cv2.polylines(display_frame, [pts], False, botcolor, 4)

                targets = bot.trajectory
                if len(targets) > 0:
                    pts = np.array(bot.trajectory, np.int32)
                    cv2.polylines(display_frame, [pts], False, (0, 0, 255), 4)
                    tar = targets[-1]
                    cv2.circle(display_frame,(int(tar[0]), int(tar[1])),20,(0,255,255), -1,)
                    cv2.putText(display_frame,str(botnum+1),(tar[0] + 20,tar[1] + 20),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=4,color = (0, 255, 255))
        

        
        cv2.putText(display_frame,"fps:"+str(int(self.fps.get_fps())),
                    (int(self.width  / 80),int(self.height / 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, 
                    thickness=4,
                    color = (255, 255, 255))
        
        
        
        cv2.putText(display_frame,"100 um",
            (int(self.width / 80),int(self.height / 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, 
            thickness=4,
            color = (255, 255, 255),
          
        )
        cv2.line(
            display_frame, 
            (int(self.width / 8),int(self.height /40)),
            (int(self.width / 8) + int(100 * (self.pix2metric)),int(self.height / 40)), 
            (255, 255, 255), 
            thickness=4
        )

        return display_frame







    def run(self):
    
        # capture from web camx
        while self._run_flag:
            self.fps.update()

            #set and read frame
            if self._play_flag == True:
                self.framenum +=1
            
            
            if self.totalnumframes !=0:
                if self.framenum >= self.totalnumframes:
                    self.framenum = 0
                
                self.cap.set(1, self.framenum)
            
            
            ret, frame = self.cap.read()
        
            #control_mask = None
            if ret:       
                if self.totalnumframes ==0:         
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                    self.pix2metric =  0.28985 * self.objective
                    

                #step 1 track robot
                self.robot_mask = self.find_robot_mask(frame)
                robotcroppedmask, recorded_cropped_frame = self.track_robot(frame, self.robot_mask) 
                

                #step 2.5: subtract robots from cell mask. first create an inital mask using cell mask params. then remove the robot from this inital mask. then find the mask again on this first mask and use this for tracking
                #create inital cell mask
           
                #on cell mask initial, replace all

           

                #step 3 display
                if True:
                    croppedmask = robotcroppedmask
                    if self.mask_flag == True:
                        frame = cv2.cvtColor(self.robot_mask, cv2.COLOR_GRAY2BGR)

                
                
                displayframe = self.display_hud(frame)

                #step 2 control robot
                
                
                #step 3: emit croppedframe, frame from this thread to the main thread
                self.cropped_frame_signal.emit(croppedmask, recorded_cropped_frame)
                self.change_pixmap_signal.emit(displayframe, self.robot_list)
            
            
                #step 4: delay based on fps
                if self.totalnumframes !=0:
                    interval = 1/self.videofps  #use original fps used to record the video if not live
                    time.sleep(interval)

    
            
           


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        #blank = np.zeros((self.width, self.height, 3), dtype=np.uint8) 
        #self.change_pixmap_signal.emit(blank)

        self._run_flag = False
        self.wait()
        self.cap.release()


