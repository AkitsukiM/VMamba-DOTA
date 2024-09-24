# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from ui_test.photoView import PhotoViewer
# from ui_test.videoPlayer import VideoPlayerView


class UI_MainWindow(object):
    def setupUi(self, MainWindow):
        # 无需看
        self.main_tab_widget = QtWidgets.QTabWidget()
        MainWindow.setCentralWidget(self.main_tab_widget)
        self.tab_widget_0 = QtWidgets.QWidget()
        # self.tab_widget_1 = QtWidgets.QWidget()
        # self.tab_widget_2 = QtWidgets.QWidget()

        self.main_tab_widget.addTab(self.tab_widget_0, "Model 0")
        # self.main_tab_widget.addTab(self.tab_widget_1, "Model 1")
        # self.main_tab_widget.addTab(self.tab_widget_2, "Model 2")

        self.tab0_UI()
        # self.tab1_UI()
        # self.tab2_UI()

        self.setup_menuUI(MainWindow)
        self.retranslateUi(MainWindow)

    def tab0_UI(self):
        main_vertical_Layout = QtWidgets.QVBoxLayout(self.tab_widget_0)
        self.widget_down0 = QtWidgets.QWidget()
        self.widget_foot0 = QtWidgets.QWidget()
        main_vertical_Layout.addWidget(self.widget_down0)
        main_vertical_Layout.addWidget(self.widget_foot0)
        down_HorizontalLayout = QtWidgets.QHBoxLayout(self.widget_down0)

        self.viewpic0_lf = PhotoViewer(self.widget_down0)
        self.viewpic0_rt = PhotoViewer(self.widget_down0)
        down_HorizontalLayout.addWidget(self.viewpic0_lf)
        down_HorizontalLayout.addWidget(self.viewpic0_rt)

        self.gridLayout = QtWidgets.QGridLayout(self.widget_foot0)
        # self.zoomout = QtWidgets.QPushButton("&放大")
        # self.zoomin = QtWidgets.QPushButton("&缩小")
        # self.zoomout.setMaximumWidth(50)
        # self.zoomin.setMaximumWidth(50)
        self.labeltext0_up = QLabel("   ")
        self.labeltext0_dn = QLabel("   ")

        for i in range(20):
            self.gridLayout.addWidget(QLabel("       "), 0, i)
        self.gridLayout.addWidget(self.labeltext0_up, 0, 1)
        self.gridLayout.addWidget(self.labeltext0_dn, 1, 1)

    #def tab1_UI(self):
    #    main_vertical_Layout = QtWidgets.QVBoxLayout(self.tab_widget_1)
    #    self.widget_down1 = QtWidgets.QWidget()
    #    self.widget_foot1 = QtWidgets.QWidget()
    #    main_vertical_Layout.addWidget(self.widget_down1)
    #    main_vertical_Layout.addWidget(self.widget_foot1)
    #    down_HorizontalLayout = QtWidgets.QHBoxLayout(self.widget_down1)

    #    self.viewpic1_lf = PhotoViewer(self.widget_down1)
    #    self.viewpic1_rt = PhotoViewer(self.widget_down1)
    #    down_HorizontalLayout.addWidget(self.viewpic1_lf)
    #    down_HorizontalLayout.addWidget(self.viewpic1_rt)

    #    self.gridLayout = QtWidgets.QGridLayout(self.widget_foot1)
    #    # self.zoomout = QtWidgets.QPushButton("&放大")
    #    # self.zoomin = QtWidgets.QPushButton("&缩小")
    #    # self.zoomout.setMaximumWidth(50)
    #    # self.zoomin.setMaximumWidth(50)
    #    self.labeltext1_up = QLabel("   ")
    #    self.labeltext1_dn = QLabel("   ")

    #    for i in range(20):
    #        self.gridLayout.addWidget(QLabel("       "), 0, i)
    #    self.gridLayout.addWidget(self.labeltext1_up, 0, 1)
    #    self.gridLayout.addWidget(self.labeltext1_dn, 1, 1)

    #def tab2_UI(self):
    #    main_vertical_Layout = QtWidgets.QVBoxLayout(self.tab_widget_2)
    #    self.widget_down2 = QtWidgets.QWidget()
    #    self.widget_foot2 = QtWidgets.QWidget()
    #    main_vertical_Layout.addWidget(self.widget_down2)
    #    main_vertical_Layout.addWidget(self.widget_foot2)
    #    down_HorizontalLayout = QtWidgets.QHBoxLayout(self.widget_down2)

    #    self.viewpic2_lf = PhotoViewer(self.widget_down2)
    #    self.viewpic2_rt = PhotoViewer(self.widget_down2)
    #    down_HorizontalLayout.addWidget(self.viewpic2_lf)
    #    down_HorizontalLayout.addWidget(self.viewpic2_rt)

    #    self.gridLayout = QtWidgets.QGridLayout(self.widget_foot2)
    #    # self.zoomout = QtWidgets.QPushButton("&放大")
    #    # self.zoomin = QtWidgets.QPushButton("&缩小")
    #    # self.zoomout.setMaximumWidth(50)
    #    # self.zoomin.setMaximumWidth(50)
    #    self.labeltext2_up = QLabel("   ")
    #    self.labeltext2_dn = QLabel("   ")

    #    for i in range(20):
    #        self.gridLayout.addWidget(QLabel("       "), 0, i)
    #    self.gridLayout.addWidget(self.labeltext2_up, 0, 1)
    #    self.gridLayout.addWidget(self.labeltext2_dn, 1, 1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "遥感目标检测算法演示软件"))
        MainWindow.setWindowIcon(QIcon('./ui_test/img/satelite.jpg'))
        # 设置窗体与桌面的大小比例关系
        desktop = QtWidgets.QApplication.desktop()
        screenHeight = desktop.height() * 0.8
        screenWidth = screenHeight * 18 / 10
        MainWindow.resize(screenWidth, screenHeight)
        # 设置最小显示大小
        MainWindow.setMinimumHeight(600)
        MainWindow.setMinimumWidth(600 * 1.8)

    def setup_menuUI(self, MainWindow):

        # 工具栏操作
        ########################################################################################################################
        tb = self.addToolBar("File")
        self.model_info = QAction(QIcon("./ui_test/img/import.png"), "模型参数", self)
        self.model_info.setStatusTip('模型参数')

        self.open_pic = QAction(QIcon("./ui_test/img/open.png"), "打开文件", self)
        self.open_pic.setStatusTip('打开文件')

        self.save_pic = QAction(QIcon("./ui_test/img/save.png"), "保存数据", self)
        self.save_pic.setStatusTip('保存数据')

        # self.set_sys_info = QAction(QIcon("./ui_test/img/run.png"), "运行", self)
        # self.run_evaluation = QAction(QIcon("./ui_test/img/run.png"), "评测", self)
        # self.run_evaluation.setStatusTip('评测')

        tb.addAction(self.model_info)
        tb.addAction(self.open_pic)
        tb.addAction(self.save_pic)
        # tb.addAction(self.run_evaluation)

        # 状态栏
        ########################################################################################################################
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
