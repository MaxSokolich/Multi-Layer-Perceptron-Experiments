
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
from classes.gui_functions import MainWindow


#  1) make sure were getting the right assignment and order of robots to simulation (should be faster one first, then slower one)
#  2) work on training and ensuring robot ends at target location
#  3) generate more data and eliminate outliers
#  4) 
#  5) chopping tracjectorys to ensure robots stay within workspace bounds

 


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
