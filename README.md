This is a repo containing the microrobotic control user interface for an algorithm for independetly controlling mulitple microrobots to unique destinations in the workspace. it uses a multi layer perceptron neural network trained via training data to determine the opimtal action commands to the magnetic manipiulation system.




4)  need to install qt5
    - sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools  
    - sudo apt install qt5-default

5) need to install Spinnaker FLIR camera SDK and python API: 
    - https://flir.app.boxcn.net/v/SpinnakerSDK/file/1093743440079
    - may need: sudo apt-mark manual qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools for spinview 

7) need to add "self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)" above self.cam.BeginAcquistion() line in $ .local/lib/python3.8/site-packages/EasyPySpin.videocapture.py

pyuic5 uis/GUI.ui -o gui_widgets.py


/opt/homebrew/bin/python3.9 -m PyQt5.uic.pyuic uis/GUI.ui -o gui_widgets.py

git reset --soft HEAD~2

![alt text](MLP-1.PNG)