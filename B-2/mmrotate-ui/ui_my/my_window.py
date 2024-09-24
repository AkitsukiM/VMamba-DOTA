# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets

from PIL import Image
from PIL.ImageQt import ImageQt

import os
import mmcv
from tqdm import tqdm

from ui_test.ui import UI_MainWindow
from ui_my.my_detection import Detection


class Func_Window1(QMainWindow, UI_MainWindow):
    def __init__(self, parent=None):
        super(Func_Window1, self).__init__(parent)
        self.setupUi(self)

        self.index = 0
        self.detect_model = None

        self.img_size_text = ''
        self.file_path_text = ''
        self.current_pil_img = None
        self.current_pil_img_detect = None

        self.qim0 = None
        self.qim0_detect = None
        self.qim1 = None
        self.qim1_detect = None
        self.qim2 = None
        self.qim2_detect = None

        self.qpm0 = None
        self.qpm0_detect = None
        self.qpm1 = None
        self.qpm1_detect = None
        self.qpm2 = None
        self.qpm2_detect = None

        self.cv_img_save_buffer = None
        self.model_info.triggered.connect(self.show_model_info)
        self.open_pic.triggered.connect(self.router_open_pic)
        self.save_pic.triggered.connect(self.save_result)


    def show_model_info(self):
        #labelTxt_dir = "data/DOTA/val/labelTxt"
        #dirs = os.listdir(labelTxt_dir)
        #for file in tqdm(dirs):
        #    for i in range(500000):
        #        pass

        self.index = self.main_tab_widget.currentIndex()
        if self.index == 0:
            print("\n\
                  Model 0 Info.\n")
        elif self.index == 1:
            print("\n\
                  Model 1 Info.\n")
        elif self.index == 2:
            print("\n\
                  Model 2 Info.\n")
        else:
            pass


    def router_open_pic(self):
        self.index = self.main_tab_widget.currentIndex()
        if self.index == 0:
            self.detect_model = Detection(method="model_0")
        elif self.index == 1:
            self.detect_model = Detection(method="model_1")
        elif self.index == 2:
            self.detect_model = Detection(method="model_2")
        else:
            pass
        self.choose_file()


    def choose_file(self):
        file_path, filetype = QFileDialog.getOpenFileName(self, "加载单个文件", "", "All Files (*)")
        if file_path == '':
            return
        suffix = os.path.splitext(os.path.split(file_path)[1])[1][1:]
        if suffix not in ['jpg', 'png']:
            self.information('读取文件格式为jpg, png')

        print(file_path)
        self.file_path_text = file_path

        current_cv_img = self.detect_model.gt_img(file_path)
        current_cv_img = self.scale_huge_img(current_cv_img, scale_limit=2048.)
        self.current_pil_img = Image.fromarray(mmcv.bgr2rgb(current_cv_img))  # change BGR(CV2-style) -> RGB(PIL-style)

        img_size = self.current_pil_img.size
        self.img_size_text = ("图片 宽 " + str(img_size[0]) + ", 高 " + str(img_size[1]))

        current_cv_img_detect = self.detect_model.predict_img(file_path)
        self.cv_img_save_buffer = current_cv_img_detect
        current_cv_img_detect = self.scale_huge_img(current_cv_img_detect, scale_limit=2048.)
        self.current_pil_img_detect = Image.fromarray(mmcv.bgr2rgb(current_cv_img_detect))  # change BGR(CV2-style) -> RGB(PIL-style)

        self.update_ui()


    def scale_huge_img(self, current_cv_img, scale_limit=2048.):

        #################################### PROCESS IMAGE OF LARGE SIZE ###################################
        # GUI can not display the image whose long or short side length > 3000 pixels (A BUG)
        # Therefore, we do not use the GUI to process this image
        # After solving this bug, we can cancel this procedure
        # project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the path of project
        # if current_pil_img.shape[0] > 3000 or current_pil_img.shape[1] > 3000:
        #     # split off-line
        #     shutil.rmtree(os.path.join(project_path, 'tmp/images_split/'))  # clear the previous temp file
        #     os.mkdir(os.path.join(project_path, 'tmp/images_split/'))
        #     split = splitbase(file_path, os.path.join(project_path, 'tmp/images_split/'),
        #                       gap=100, subsize=1024, ext=file_path.split('.')[1])
        #     split.SplitSingle(file_path.split('.')[0], 0.25, extent=file_path.split('.')[1])
        #     self.detect_model.predict_imgs(os.path.join(project_path, 'tmp/images_split/'), file_path)
        #
        #     return
        ####################################################################################################

        w, h, _ = current_cv_img.shape
        if w > scale_limit or h > scale_limit:
            scale_factor = min(scale_limit / w, scale_limit / h)
            scale_img = mmcv.imrescale(current_cv_img, scale_factor)
            return scale_img
        else:
            return current_cv_img
        

    def update_ui(self):
        if self.index == 0:
            self.labeltext0_up.setText(self.img_size_text)
            self.labeltext0_dn.setText(self.file_path_text)

            self.qim0 = ImageQt(self.current_pil_img)
            self.qpm0 = QtGui.QPixmap.fromImage(self.qim0)
            self.viewpic0_lf.setPhoto(self.qpm0)

            self.qim0_detect = ImageQt(self.current_pil_img_detect)
            self.qpm0_detect = QtGui.QPixmap.fromImage(self.qim0_detect)
            self.viewpic0_rt.setPhoto(self.qpm0_detect)

        elif self.index == 1:
            self.labeltext1_up.setText(self.img_size_text)
            self.labeltext1_dn.setText(self.file_path_text)

            self.qim1 = ImageQt(self.current_pil_img)
            self.qpm1 = QtGui.QPixmap.fromImage(self.qim1)
            self.viewpic1_lf.setPhoto(self.qpm1)

            self.qim1_detect = ImageQt(self.current_pil_img_detect)
            self.qpm1_detect = QtGui.QPixmap.fromImage(self.qim1_detect)
            self.viewpic1_rt.setPhoto(self.qpm1_detect)

        elif self.index == 2:
            self.labeltext2_up.setText(self.img_size_text)
            self.labeltext2_dn.setText(self.file_path_text)

            self.qim2 = ImageQt(self.current_pil_img)
            self.qpm2 = QtGui.QPixmap.fromImage(self.qim2)
            self.viewpic2_lf.setPhoto(self.qpm2)

            self.qim2_detect = ImageQt(self.current_pil_img_detect)
            self.qpm2_detect = QtGui.QPixmap.fromImage(self.qim2_detect)
            self.viewpic2_rt.setPhoto(self.qpm2_detect)
            
        else:
            pass


    def save_result(self):
        if self.cv_img_save_buffer is None:
            self.information('请生成图像！')
            return
        jpg_path, ok = QFileDialog.getSaveFileName(self, "图像保存", "", "All Files (*)")
        if not jpg_path:
            return
        suffix = os.path.splitext(os.path.split(jpg_path)[1])[1][1:]

        if suffix not in ['jpg', 'png']:
            self.information('保存文件格式为jpg, png')
            return
        mmcv.imwrite(self.cv_img_save_buffer, jpg_path)
        print("Save to", jpg_path, "successfully!")

    
    def information(self, msg):
        QMessageBox.information(self, 'Message', msg, QMessageBox.Yes)

