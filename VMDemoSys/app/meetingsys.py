import json
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import cv2
from PyQt5.QtCore import QTimer
import os
from enum import Enum
import numpy as np

from ..core.matter import Matter


class Controller(Enum):
    TIMER = 1
    SLIDER = 2


class MeetingSystem(QWidget):

    def __init__(self, args):
        super(MeetingSystem, self).__init__()
        self.cap = None
        ref_img_path=args.ref
        self.matter = Matter()
        self.matter.set_tgt(ref_img_path)
        self.timer_camera = QTimer()
        

        pos_unit = 90

        # 外框
        # self.resize(900, 650)
        self.resize(1366, 960)
        self.setWindowTitle("Video Meeting System")
        # 图片label
        self.im_label = QLabel(self)
        self.im_label.setText("Waiting for Frame...")
        self.im_label.setFixedSize(1240, 720)  # width height
        self.im_label.move(0, 0)
        self.im_label.setStyleSheet("QLabel{background:gray;}"
                                    "QLabel{color:rgb(100,255,100);font-size:15px;font-weight:bold;font-family:SimSun;}"
                                    )
        # START MEETING BUTTON
        self.btn_start_meeting = QPushButton(self)
        self.btn_start_meeting.setText("Start meeting")
        self.btn_start_meeting.move(5*pos_unit+400, 740)
        self.btn_start_meeting.setStyleSheet(
            "QPushButton{background:yellow;}")  # 没检测红色，检测绿色
        self.btn_start_meeting.clicked.connect(self.start_meeting)

        # START MEETING BUTTON
        self.btn_end_meeting = QPushButton(self)
        self.btn_end_meeting.setText("End meeting")
        self.btn_end_meeting.move(5*pos_unit+500, 740)
        self.btn_end_meeting.setStyleSheet(
            "QPushButton{background:blue;color:rgb(255,255,255)}")  # 没检测红色，检测绿色
        self.btn_end_meeting.clicked.connect(self.end_meeting)

    def start_meeting(self):
        self.cap = cv2.VideoCapture(0)
        self.timer_camera.start(20)
        self.timer_camera.timeout.connect(self.readFrame)
        self.frame_idx=0

    def end_meeting(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.timer_camera.stop()
        self.im_label.setText("This video has been stopped.")
        self.im_label.setStyleSheet("QLabel{background:gray;}"
                                    "QLabel{color:rgb(100,255,100);font-size:15px;font-weight:bold;font-family:SimSun;}"
                                    )

    def readFrame(self):
        ret, frame = self.cap.read()
        
        if not ret:
            print('Read frame failed!!!')
            exit()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        frame_new = self.matter.run(frame)
        cv2.imwrite('/Users/jiangxin/Project/VBMH/vbmh_tgt_filter/VMDemoSys/data/'+str(10000+self.frame_idx)+'.png',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        cv2.imwrite('/Users/jiangxin/Project/VBMH/vbmh_tgt_filter/VMDemoSys/data/'+str(20000+self.frame_idx)+'.png',cv2.cvtColor(frame_new,cv2.COLOR_RGB2BGR))
        frame=np.concatenate([frame,frame_new],axis=1)
        
        height, width, bytesPerComponent = frame.shape
        bytesPerLine = bytesPerComponent * width
        q_image = QImage(frame.data,  width, height, bytesPerLine,
                         QImage.Format_RGB888).scaled(self.im_label.width(), self.im_label.height())
        self.im_label.setPixmap(QPixmap.fromImage(q_image))
        self.frame_idx+=1


def start_meeting(args):
    app = QtWidgets.QApplication(sys.argv)
    my = MeetingSystem(args)
    my.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    my = MeetingSystem()
    my.show()
    sys.exit(app.exec_())
