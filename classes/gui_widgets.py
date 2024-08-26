# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'uis/GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1402, 1014)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("font-size: 12pt; font: Arial;")
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(10, 850, 1041, 131))
        self.plainTextEdit.setMouseTracking(True)
        self.plainTextEdit.setStyleSheet("font-size: 15pt; font: Arial;")
        self.plainTextEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.plainTextEdit.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.plainTextEdit.setPlainText("")
        self.plainTextEdit.setCenterOnScroll(False)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.VideoFeedLabel = QtWidgets.QLabel(self.centralwidget)
        self.VideoFeedLabel.setGeometry(QtCore.QRect(10, 5, 1041, 801))
        self.VideoFeedLabel.setMouseTracking(True)
        self.VideoFeedLabel.setStyleSheet("background-color: rgb(0,0,0); border:2px solid rgb(255, 0, 0); ")
        self.VideoFeedLabel.setText("")
        self.VideoFeedLabel.setObjectName("VideoFeedLabel")
        self.frameslider = QtWidgets.QProgressBar(self.centralwidget)
        self.frameslider.setGeometry(QtCore.QRect(10, 812, 1041, 31))
        self.frameslider.setStyleSheet("    QProgressBar {\n"
"        border: 2px solid rgba(33, 37, 43, 180);\n"
"        border-radius: 5px;\n"
"        text-align: center;\n"
"        background-color: rgba(33, 37, 43, 180);\n"
"        color: black;\n"
"    }\n"
"    QProgressBar::chunk {\n"
"        background-color: #FFD700;\n"
"    }")
        self.frameslider.setMinimum(0)
        self.frameslider.setMaximum(100)
        self.frameslider.setProperty("value", 0)
        self.frameslider.setObjectName("frameslider")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setMinimumSize(QtCore.QSize(329, 987))
        self.dockWidget.setStyleSheet("")
        self.dockWidget.setFloating(False)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.frame_3 = QtWidgets.QFrame(self.dockWidgetContents)
        self.frame_3.setGeometry(QtCore.QRect(10, 10, 311, 231))
        self.frame_3.setStyleSheet(" color: rgb(0, 0, 0);\n"
" background-color: rgb(255, 255, 255);\n"
"")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.savedatabutton = QtWidgets.QPushButton(self.frame_3)
        self.savedatabutton.setGeometry(QtCore.QRect(10, 140, 81, 31))
        self.savedatabutton.setStyleSheet("QPushButton {\n"
"                color: rgb(0, 0, 0);\n"
"                background-color: rgb(255, 255, 0);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 255, 100);\n"
"                min-width: 1em;\n"
"                padding: 6px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(255, 255, 200);\n"
"                color: rgb(0, 0, 0);\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: red;\n"
"                border: 2px solid red;\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"                border-style: inset;\n"
"                }")
        self.savedatabutton.setCheckable(True)
        self.savedatabutton.setObjectName("savedatabutton")
        self.recordbutton = QtWidgets.QPushButton(self.frame_3)
        self.recordbutton.setGeometry(QtCore.QRect(100, 140, 71, 31))
        self.recordbutton.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 0);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 0);\n"
"                min-width: 1em;\n"
"                padding: 6px;\n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 16px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(100, 100, 100);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.recordbutton.setCheckable(True)
        self.recordbutton.setObjectName("recordbutton")
        self.framelabel = QtWidgets.QLabel(self.frame_3)
        self.framelabel.setGeometry(QtCore.QRect(10, 120, 121, 21))
        self.framelabel.setMaximumSize(QtCore.QSize(300, 25))
        self.framelabel.setObjectName("framelabel")
        self.trackbutton = QtWidgets.QPushButton(self.frame_3)
        self.trackbutton.setGeometry(QtCore.QRect(80, 5, 141, 31))
        self.trackbutton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.trackbutton.setStyleSheet("\n"
"QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 255);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 255);\n"
"                min-width: 1em;\n"
"                padding: 2px;\n"
"              font: bold 15px;\n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 25px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(0, 255, 0);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.trackbutton.setCheckable(True)
        self.trackbutton.setObjectName("trackbutton")
        self.exposurelabel = QtWidgets.QLabel(self.frame_3)
        self.exposurelabel.setGeometry(QtCore.QRect(20, 170, 111, 25))
        self.exposurelabel.setMaximumSize(QtCore.QSize(150, 25))
        self.exposurelabel.setObjectName("exposurelabel")
        self.exposurebox = QtWidgets.QSpinBox(self.frame_3)
        self.exposurebox.setGeometry(QtCore.QRect(20, 190, 111, 35))
        self.exposurebox.setMinimum(100)
        self.exposurebox.setMaximum(30000)
        self.exposurebox.setSingleStep(100)
        self.exposurebox.setProperty("value", 5000)
        self.exposurebox.setDisplayIntegerBase(10)
        self.exposurebox.setObjectName("exposurebox")
        self.objectivelabel = QtWidgets.QLabel(self.frame_3)
        self.objectivelabel.setGeometry(QtCore.QRect(180, 170, 111, 25))
        self.objectivelabel.setMaximumSize(QtCore.QSize(150, 25))
        self.objectivelabel.setObjectName("objectivelabel")
        self.objectivebox = QtWidgets.QSpinBox(self.frame_3)
        self.objectivebox.setGeometry(QtCore.QRect(180, 190, 111, 35))
        self.objectivebox.setMinimum(1)
        self.objectivebox.setMaximum(50)
        self.objectivebox.setSingleStep(5)
        self.objectivebox.setProperty("value", 10)
        self.objectivebox.setDisplayIntegerBase(10)
        self.objectivebox.setObjectName("objectivebox")
        self.algorithbutton = QtWidgets.QPushButton(self.frame_3)
        self.algorithbutton.setGeometry(QtCore.QRect(70, 85, 161, 31))
        self.algorithbutton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.algorithbutton.setStyleSheet("\n"
"QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 255);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 255);\n"
"                min-width: 1em;\n"
"                padding: 2px;\n"
"              font: bold 15px;\n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 25px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(0, 255, 0);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.algorithbutton.setCheckable(True)
        self.algorithbutton.setObjectName("algorithbutton")
        self.generatepathbutton = QtWidgets.QPushButton(self.frame_3)
        self.generatepathbutton.setGeometry(QtCore.QRect(50, 45, 201, 31))
        self.generatepathbutton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.generatepathbutton.setStyleSheet("\n"
"QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 255);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 255);\n"
"                min-width: 1em;\n"
"                padding: 2px;\n"
"              font: bold 15px;\n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 25px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(0, 255, 0);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.generatepathbutton.setCheckable(False)
        self.generatepathbutton.setObjectName("generatepathbutton")
        self.choosevideobutton = QtWidgets.QPushButton(self.frame_3)
        self.choosevideobutton.setGeometry(QtCore.QRect(190, 140, 101, 31))
        self.choosevideobutton.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 0);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 0);\n"
"                min-width: 1em;\n"
"                padding: 6px;\n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 16px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(100, 100, 100);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.choosevideobutton.setCheckable(False)
        self.choosevideobutton.setObjectName("choosevideobutton")
        self.trackerparamsframe = QtWidgets.QFrame(self.dockWidgetContents)
        self.trackerparamsframe.setGeometry(QtCore.QRect(10, 250, 311, 281))
        self.trackerparamsframe.setStyleSheet(" color: rgb(0, 0, 0);\n"
" background-color: rgb(255, 255, 255);\n"
"")
        self.trackerparamsframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.trackerparamsframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.trackerparamsframe.setObjectName("trackerparamsframe")
        self.robotmasklowerbox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.robotmasklowerbox.setGeometry(QtCore.QRect(110, 175, 61, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.robotmasklowerbox.setFont(font)
        self.robotmasklowerbox.setStyleSheet("")
        self.robotmasklowerbox.setMaximum(255)
        self.robotmasklowerbox.setSingleStep(1)
        self.robotmasklowerbox.setProperty("value", 0)
        self.robotmasklowerbox.setObjectName("robotmasklowerbox")
        self.maskthreshlabel = QtWidgets.QLabel(self.trackerparamsframe)
        self.maskthreshlabel.setGeometry(QtCore.QRect(10, 170, 81, 25))
        self.maskthreshlabel.setMaximumSize(QtCore.QSize(150, 25))
        self.maskthreshlabel.setObjectName("maskthreshlabel")
        self.robotmaskdilationbox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.robotmaskdilationbox.setGeometry(QtCore.QRect(110, 235, 61, 20))
        self.robotmaskdilationbox.setMaximum(40)
        self.robotmaskdilationbox.setObjectName("robotmaskdilationbox")
        self.maskdilationlabel = QtWidgets.QLabel(self.trackerparamsframe)
        self.maskdilationlabel.setGeometry(QtCore.QRect(10, 230, 81, 25))
        self.maskdilationlabel.setMaximumSize(QtCore.QSize(150, 25))
        self.maskdilationlabel.setObjectName("maskdilationlabel")
        self.robotcroplengthbox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.robotcroplengthbox.setGeometry(QtCore.QRect(110, 115, 61, 21))
        self.robotcroplengthbox.setMinimum(5)
        self.robotcroplengthbox.setMaximum(400)
        self.robotcroplengthbox.setSingleStep(1)
        self.robotcroplengthbox.setProperty("value", 40)
        self.robotcroplengthbox.setDisplayIntegerBase(10)
        self.robotcroplengthbox.setObjectName("robotcroplengthbox")
        self.croplengthlabel = QtWidgets.QLabel(self.trackerparamsframe)
        self.croplengthlabel.setGeometry(QtCore.QRect(10, 110, 81, 25))
        self.croplengthlabel.setMaximumSize(QtCore.QSize(150, 25))
        self.croplengthlabel.setObjectName("croplengthlabel")
        self.robotmaskblurbox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.robotmaskblurbox.setGeometry(QtCore.QRect(110, 145, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.robotmaskblurbox.setFont(font)
        self.robotmaskblurbox.setStyleSheet("")
        self.robotmaskblurbox.setMaximum(40)
        self.robotmaskblurbox.setSingleStep(1)
        self.robotmaskblurbox.setProperty("value", 0)
        self.robotmaskblurbox.setObjectName("robotmaskblurbox")
        self.maskblurlabel = QtWidgets.QLabel(self.trackerparamsframe)
        self.maskblurlabel.setGeometry(QtCore.QRect(10, 140, 61, 25))
        self.maskblurlabel.setMaximumSize(QtCore.QSize(150, 25))
        self.maskblurlabel.setObjectName("maskblurlabel")
        self.maskbutton = QtWidgets.QPushButton(self.trackerparamsframe)
        self.maskbutton.setGeometry(QtCore.QRect(20, 10, 101, 21))
        self.maskbutton.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 0);\n"
"                border-style: outset;\n"
"                border-width: 2px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 0);\n"
"                min-width: 1em;\n"
"                padding: 1px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-style: inset;\n"
"              border-color: rgb(0, 255, 0);\n"
"            }")
        self.maskbutton.setCheckable(True)
        self.maskbutton.setChecked(False)
        self.maskbutton.setObjectName("maskbutton")
        self.maskinvert_checkBox = QtWidgets.QCheckBox(self.trackerparamsframe)
        self.maskinvert_checkBox.setGeometry(QtCore.QRect(140, 10, 131, 21))
        self.maskinvert_checkBox.setChecked(False)
        self.maskinvert_checkBox.setObjectName("maskinvert_checkBox")
        self.robotmaskupperbox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.robotmaskupperbox.setGeometry(QtCore.QRect(110, 205, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.robotmaskupperbox.setFont(font)
        self.robotmaskupperbox.setStyleSheet("")
        self.robotmaskupperbox.setMinimum(0)
        self.robotmaskupperbox.setMaximum(255)
        self.robotmaskupperbox.setSingleStep(1)
        self.robotmaskupperbox.setProperty("value", 128)
        self.robotmaskupperbox.setObjectName("robotmaskupperbox")
        self.maskthreshlabel_2 = QtWidgets.QLabel(self.trackerparamsframe)
        self.maskthreshlabel_2.setGeometry(QtCore.QRect(10, 200, 81, 25))
        self.maskthreshlabel_2.setMaximumSize(QtCore.QSize(150, 25))
        self.maskthreshlabel_2.setObjectName("maskthreshlabel_2")
        self.numobstaclelabel = QtWidgets.QLabel(self.trackerparamsframe)
        self.numobstaclelabel.setGeometry(QtCore.QRect(200, 30, 101, 25))
        self.numobstaclelabel.setMaximumSize(QtCore.QSize(150, 25))
        self.numobstaclelabel.setObjectName("numobstaclelabel")
        self.safteyradius = QtWidgets.QLabel(self.trackerparamsframe)
        self.safteyradius.setGeometry(QtCore.QRect(10, 50, 81, 25))
        self.safteyradius.setMaximumSize(QtCore.QSize(150, 25))
        self.safteyradius.setObjectName("safteyradius")
        self.alphabox = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.alphabox.setGeometry(QtCore.QRect(110, 85, 61, 21))
        self.alphabox.setMinimum(0)
        self.alphabox.setMaximum(10)
        self.alphabox.setSingleStep(1)
        self.alphabox.setProperty("value", 2)
        self.alphabox.setDisplayIntegerBase(10)
        self.alphabox.setObjectName("alphabox")
        self.alpha = QtWidgets.QLabel(self.trackerparamsframe)
        self.alpha.setGeometry(QtCore.QRect(10, 80, 81, 25))
        self.alpha.setMaximumSize(QtCore.QSize(150, 25))
        self.alpha.setObjectName("alpha")
        self.safetyradiusbox = QtWidgets.QDoubleSpinBox(self.trackerparamsframe)
        self.safetyradiusbox.setGeometry(QtCore.QRect(110, 50, 61, 22))
        self.safetyradiusbox.setMaximum(1.0)
        self.safetyradiusbox.setProperty("value", 0.5)
        self.safetyradiusbox.setObjectName("safetyradiusbox")
        self.delta_1 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_1.setGeometry(QtCore.QRect(220, 60, 61, 21))
        self.delta_1.setMinimum(0)
        self.delta_1.setMaximum(100)
        self.delta_1.setSingleStep(1)
        self.delta_1.setProperty("value", 0)
        self.delta_1.setDisplayIntegerBase(10)
        self.delta_1.setObjectName("delta_1")
        self.delta_2 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_2.setGeometry(QtCore.QRect(220, 80, 61, 21))
        self.delta_2.setMinimum(0)
        self.delta_2.setMaximum(100)
        self.delta_2.setSingleStep(1)
        self.delta_2.setProperty("value", 0)
        self.delta_2.setDisplayIntegerBase(10)
        self.delta_2.setObjectName("delta_2")
        self.delta_3 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_3.setGeometry(QtCore.QRect(220, 100, 61, 21))
        self.delta_3.setMinimum(0)
        self.delta_3.setMaximum(100)
        self.delta_3.setSingleStep(1)
        self.delta_3.setProperty("value", 0)
        self.delta_3.setDisplayIntegerBase(10)
        self.delta_3.setObjectName("delta_3")
        self.delta_4 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_4.setGeometry(QtCore.QRect(220, 120, 61, 21))
        self.delta_4.setMinimum(0)
        self.delta_4.setMaximum(100)
        self.delta_4.setSingleStep(1)
        self.delta_4.setProperty("value", 0)
        self.delta_4.setDisplayIntegerBase(10)
        self.delta_4.setObjectName("delta_4")
        self.delta_8 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_8.setGeometry(QtCore.QRect(220, 200, 61, 21))
        self.delta_8.setMinimum(0)
        self.delta_8.setMaximum(100)
        self.delta_8.setSingleStep(1)
        self.delta_8.setProperty("value", 0)
        self.delta_8.setDisplayIntegerBase(10)
        self.delta_8.setObjectName("delta_8")
        self.delta_6 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_6.setGeometry(QtCore.QRect(220, 160, 61, 21))
        self.delta_6.setMinimum(0)
        self.delta_6.setMaximum(100)
        self.delta_6.setSingleStep(1)
        self.delta_6.setProperty("value", 0)
        self.delta_6.setDisplayIntegerBase(10)
        self.delta_6.setObjectName("delta_6")
        self.delta_7 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_7.setGeometry(QtCore.QRect(220, 180, 61, 21))
        self.delta_7.setMinimum(0)
        self.delta_7.setMaximum(100)
        self.delta_7.setSingleStep(1)
        self.delta_7.setProperty("value", 0)
        self.delta_7.setDisplayIntegerBase(10)
        self.delta_7.setObjectName("delta_7")
        self.delta_5 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_5.setGeometry(QtCore.QRect(220, 140, 61, 21))
        self.delta_5.setMinimum(0)
        self.delta_5.setMaximum(100)
        self.delta_5.setSingleStep(1)
        self.delta_5.setProperty("value", 0)
        self.delta_5.setDisplayIntegerBase(10)
        self.delta_5.setObjectName("delta_5")
        self.delta_10 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_10.setGeometry(QtCore.QRect(220, 240, 61, 21))
        self.delta_10.setMinimum(0)
        self.delta_10.setMaximum(100)
        self.delta_10.setSingleStep(1)
        self.delta_10.setProperty("value", 0)
        self.delta_10.setDisplayIntegerBase(10)
        self.delta_10.setObjectName("delta_10")
        self.delta_9 = QtWidgets.QSpinBox(self.trackerparamsframe)
        self.delta_9.setGeometry(QtCore.QRect(220, 220, 61, 21))
        self.delta_9.setMinimum(0)
        self.delta_9.setMaximum(100)
        self.delta_9.setSingleStep(1)
        self.delta_9.setProperty("value", 0)
        self.delta_9.setDisplayIntegerBase(10)
        self.delta_9.setObjectName("delta_9")
        self.robotparamsframe = QtWidgets.QFrame(self.dockWidgetContents)
        self.robotparamsframe.setGeometry(QtCore.QRect(10, 549, 311, 61))
        self.robotparamsframe.setStyleSheet(" color: rgb(255, 255, 255);\n"
" background-color: rgb(0, 0, 0);\n"
"font-size: 14pt; font: Arial;")
        self.robotparamsframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.robotparamsframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.robotparamsframe.setObjectName("robotparamsframe")
        self.robotsizelabel = QtWidgets.QLabel(self.robotparamsframe)
        self.robotsizelabel.setGeometry(QtCore.QRect(30, 0, 51, 20))
        self.robotsizelabel.setMaximumSize(QtCore.QSize(300, 50))
        self.robotsizelabel.setObjectName("robotsizelabel")
        self.robotvelocitylabel = QtWidgets.QLabel(self.robotparamsframe)
        self.robotvelocitylabel.setGeometry(QtCore.QRect(110, 0, 81, 21))
        self.robotvelocitylabel.setMaximumSize(QtCore.QSize(300, 50))
        self.robotvelocitylabel.setObjectName("robotvelocitylabel")
        self.robotblurlabel = QtWidgets.QLabel(self.robotparamsframe)
        self.robotblurlabel.setGeometry(QtCore.QRect(220, 0, 51, 20))
        self.robotblurlabel.setMaximumSize(QtCore.QSize(16777215, 50))
        self.robotblurlabel.setStyleSheet("")
        self.robotblurlabel.setObjectName("robotblurlabel")
        self.blurlcdnum = QtWidgets.QLCDNumber(self.robotparamsframe)
        self.blurlcdnum.setGeometry(QtCore.QRect(200, 20, 61, 30))
        self.blurlcdnum.setStyleSheet("background-color: rgb(0,0,0); \n"
"color: rgb(0,255,0);")
        self.blurlcdnum.setLineWidth(0)
        self.blurlcdnum.setMidLineWidth(0)
        self.blurlcdnum.setSmallDecimalPoint(False)
        self.blurlcdnum.setDigitCount(3)
        self.blurlcdnum.setMode(QtWidgets.QLCDNumber.Dec)
        self.blurlcdnum.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.blurlcdnum.setProperty("value", 137.0)
        self.blurlcdnum.setObjectName("blurlcdnum")
        self.robotvelocityunitslabel = QtWidgets.QLabel(self.robotparamsframe)
        self.robotvelocityunitslabel.setGeometry(QtCore.QRect(160, 30, 41, 20))
        self.robotvelocityunitslabel.setMaximumSize(QtCore.QSize(300, 50))
        self.robotvelocityunitslabel.setObjectName("robotvelocityunitslabel")
        self.robotblurunitslabe = QtWidgets.QLabel(self.robotparamsframe)
        self.robotblurunitslabe.setGeometry(QtCore.QRect(260, 30, 41, 20))
        self.robotblurunitslabe.setMaximumSize(QtCore.QSize(300, 50))
        self.robotblurunitslabe.setObjectName("robotblurunitslabe")
        self.sizelcdnum = QtWidgets.QLCDNumber(self.robotparamsframe)
        self.sizelcdnum.setGeometry(QtCore.QRect(0, 20, 61, 30))
        self.sizelcdnum.setStyleSheet("background-color: rgb(0,0,0); \n"
"color: rgb(0,255,0);")
        self.sizelcdnum.setLineWidth(0)
        self.sizelcdnum.setMidLineWidth(0)
        self.sizelcdnum.setSmallDecimalPoint(False)
        self.sizelcdnum.setDigitCount(3)
        self.sizelcdnum.setMode(QtWidgets.QLCDNumber.Dec)
        self.sizelcdnum.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.sizelcdnum.setProperty("value", 20.0)
        self.sizelcdnum.setObjectName("sizelcdnum")
        self.vellcdnum = QtWidgets.QLCDNumber(self.robotparamsframe)
        self.vellcdnum.setGeometry(QtCore.QRect(90, 20, 71, 30))
        self.vellcdnum.setStyleSheet("background-color: rgb(0,0,0); \n"
"color: rgb(0,255,0);")
        self.vellcdnum.setLineWidth(0)
        self.vellcdnum.setMidLineWidth(0)
        self.vellcdnum.setSmallDecimalPoint(False)
        self.vellcdnum.setDigitCount(3)
        self.vellcdnum.setMode(QtWidgets.QLCDNumber.Dec)
        self.vellcdnum.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.vellcdnum.setProperty("value", 17.1)
        self.vellcdnum.setObjectName("vellcdnum")
        self.robotsizeunitslabel = QtWidgets.QLabel(self.robotparamsframe)
        self.robotsizeunitslabel.setGeometry(QtCore.QRect(60, 30, 31, 20))
        self.robotsizeunitslabel.setMaximumSize(QtCore.QSize(300, 50))
        self.robotsizeunitslabel.setObjectName("robotsizeunitslabel")
        self.CroppedVideoFeedLabel = QtWidgets.QLabel(self.dockWidgetContents)
        self.CroppedVideoFeedLabel.setGeometry(QtCore.QRect(10, 615, 310, 310))
        self.CroppedVideoFeedLabel.setStyleSheet("background-color: rgb(0,0,0); border:2px solid rgb(255, 0, 0); ")
        self.CroppedVideoFeedLabel.setText("")
        self.CroppedVideoFeedLabel.setObjectName("CroppedVideoFeedLabel")
        self.resetdefaultbutton = QtWidgets.QPushButton(self.dockWidgetContents)
        self.resetdefaultbutton.setGeometry(QtCore.QRect(20, 935, 71, 25))
        self.resetdefaultbutton.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.resetdefaultbutton.setFont(font)
        self.resetdefaultbutton.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(100, 100, 100);\n"
"                border-style: outset;\n"
"                border-width: 2px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(100, 100, 100);\n"
"                min-width: 1em;\n"
"                padding: 1px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(200, 200, 200);\n"
"                color: rgb(0, 0, 0);\n"
"            }\n"
"            QPushButton:pressed {\n"
"                background-color: rgb(200, 200, 200);\n"
"         \n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"                border-style: inset;\n"
"                }")
        self.resetdefaultbutton.setObjectName("resetdefaultbutton")
        self.croppedmasktoggle = QtWidgets.QPushButton(self.dockWidgetContents)
        self.croppedmasktoggle.setGeometry(QtCore.QRect(125, 935, 71, 25))
        self.croppedmasktoggle.setMaximumSize(QtCore.QSize(300, 25))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.croppedmasktoggle.setFont(font)
        self.croppedmasktoggle.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 0);\n"
"                border-style: outset;\n"
"                border-width: 2px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 0);\n"
"                min-width: 1em;\n"
"                padding: 1px;\n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-style: inset;\n"
"              border-color: rgb(0, 255, 0);\n"
"            }")
        self.croppedmasktoggle.setCheckable(True)
        self.croppedmasktoggle.setObjectName("croppedmasktoggle")
        self.croppedrecordbutton = QtWidgets.QPushButton(self.dockWidgetContents)
        self.croppedrecordbutton.setGeometry(QtCore.QRect(230, 935, 71, 25))
        self.croppedrecordbutton.setMinimumSize(QtCore.QSize(21, 0))
        self.croppedrecordbutton.setStyleSheet("QPushButton {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(0, 0, 0);\n"
"                border-style: outset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(0, 0, 0);\n"
"                min-width: 1em;\n"
"      \n"
"            }\n"
"      \n"
"            QPushButton:checked {\n"
"                color: rgb(255, 255, 255);\n"
"                background-color: rgb(255, 0, 0);\n"
"                border-style: inset;\n"
"                border-width: 3px;\n"
"                border-radius: 10px;\n"
"                border-color: rgb(255, 0, 0);\n"
"                font: bold 16px;\n"
"                min-width: 1em;\n"
"               \n"
"            }\n"
"            QPushButton:hover {\n"
"                background-color: rgb(100, 100, 100);\n"
"                color: rgb(255, 255, 255);\n"
"                border-color: rgb(100, 100, 100);\n"
"                padding-left: 5px;\n"
"                padding-top: 5px;\n"
"            }")
        self.croppedrecordbutton.setCheckable(True)
        self.croppedrecordbutton.setObjectName("croppedrecordbutton")
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.actiondock = QtWidgets.QAction(MainWindow)
        self.actiondock.setMenuRole(QtWidgets.QAction.NoRole)
        self.actiondock.setObjectName("actiondock")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.frameslider.setFormat(_translate("MainWindow", "Frame %v"))
        self.savedatabutton.setText(_translate("MainWindow", "Save Data"))
        self.recordbutton.setText(_translate("MainWindow", "Record"))
        self.framelabel.setText(_translate("MainWindow", "Frame: "))
        self.trackbutton.setText(_translate("MainWindow", "Track"))
        self.exposurelabel.setText(_translate("MainWindow", "Exposure"))
        self.objectivelabel.setText(_translate("MainWindow", "Objective"))
        self.algorithbutton.setText(_translate("MainWindow", "Apply Algorithm"))
        self.generatepathbutton.setText(_translate("MainWindow", "Generate Path"))
        self.choosevideobutton.setText(_translate("MainWindow", "Choose Video"))
        self.maskthreshlabel.setText(_translate("MainWindow", "Lower Thresh"))
        self.maskdilationlabel.setText(_translate("MainWindow", "Dilation"))
        self.croplengthlabel.setText(_translate("MainWindow", "Crop Length"))
        self.maskblurlabel.setText(_translate("MainWindow", "Blur"))
        self.maskbutton.setText(_translate("MainWindow", "Mask"))
        self.maskinvert_checkBox.setText(_translate("MainWindow", "Invert Mask: True"))
        self.maskthreshlabel_2.setText(_translate("MainWindow", "Upper Thresh"))
        self.numobstaclelabel.setText(_translate("MainWindow", "Obstacle Number"))
        self.safteyradius.setText(_translate("MainWindow", "Safety Radius"))
        self.alpha.setText(_translate("MainWindow", "Alpha"))
        self.robotsizelabel.setText(_translate("MainWindow", "Size:   "))
        self.robotvelocitylabel.setText(_translate("MainWindow", "Velocity: "))
        self.robotblurlabel.setText(_translate("MainWindow", "Blur:"))
        self.robotvelocityunitslabel.setText(_translate("MainWindow", "um/s"))
        self.robotblurunitslabe.setText(_translate("MainWindow", "units"))
        self.robotsizeunitslabel.setText(_translate("MainWindow", "um  "))
        self.resetdefaultbutton.setText(_translate("MainWindow", "Defaults"))
        self.croppedmasktoggle.setText(_translate("MainWindow", "Original"))
        self.croppedrecordbutton.setText(_translate("MainWindow", "Record"))
        self.actiondock.setText(_translate("MainWindow", "dock"))