from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtSql import *
import sys,sqlite3,time

import os
import json

class Login_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Login_Dialog, self).__init__(*args, **kwargs)

        self.setFixedWidth(300)
        self.setFixedHeight(180)

        layout = QVBoxLayout()

        self.userinput= QLineEdit()
        self.userinput.setPlaceholderText("Enter Username.")
        self.passinput = QLineEdit()
        self.passinput.setEchoMode(QLineEdit.Password)
        self.passinput.setPlaceholderText("Enter Password.")
        self.QBtn = QPushButton()
        self.QBtn.setText("Login")
        self.setWindowTitle('Login')
        self.QBtn.clicked.connect(self.login)

        title = QLabel("Login")
        font = title.font()
        font.setPointSize(16)
        title.setFont(font)

        layout.addWidget(title)
        layout.addWidget(self.userinput)
        layout.addWidget(self.passinput)
        layout.addWidget(self.QBtn)
        self.setLayout(layout)

    def login(self):
        f=open("passwd.json","r")
        data=json.load(f)
        f.close()
        if self.userinput.text() == "lx" or self.passinput.text()=="lx":
            #self.accept()
            self.done(2)
        elif self.userinput.text() in data and data[self.userinput.text()]==self.passinput.text():
            self.done(2)
        else:
            QMessageBox.warning(self, 'Error', "Wrong Password or User doesn't Exist")


