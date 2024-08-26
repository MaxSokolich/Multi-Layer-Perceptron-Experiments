This is a repo containing the microrobotic control user interface for a novel geometry based path planning obstacle avoidance algorithm.




4)  need to install qt5
    - sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools  
    - sudo apt install qt5-default

5) need to install Spinnaker FLIR camera SDK and python API: 
    - https://flir.app.boxcn.net/v/SpinnakerSDK/file/1093743440079
    - may need: sudo apt-mark manual qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools for spinview 

7) need to add "self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)" above self.cam.BeginAcquistion() line in $ .local/lib/python3.8/site-packages/EasyPySpin.videocapture.py




pyuic5 uis/GUI.ui -o gui_widgets.py


/opt/homebrew/bin/python3.9 -m PyQt5.uic.pyuic uis/GUI.ui -o gui_widgets.py
<img width="1726" alt="GUI" src="https://github.com/user-attachments/assets/405738fd-477d-40c1-ba8c-38b9571400e5">

