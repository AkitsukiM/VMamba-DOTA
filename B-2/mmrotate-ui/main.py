# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import *

from ui_my.my_window import Func_Window1 as Window
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = Window()
    myWin.show()
    sys.exit(app.exec_())
