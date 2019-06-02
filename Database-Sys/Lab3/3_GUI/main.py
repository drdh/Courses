from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
#from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtPrintSupport import *
from PyQt5.QtSql import *
import sys,sqlite3,time

import os


class Branch_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Branch_Dialog, self).__init__(*args, **kwargs)  
        #ui
        self.tableWidget = QTableView()

        self.db=QSqlDatabase.addDatabase('QSQLITE')
        self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('branch')
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)
        
        self.model.setHeaderData(0, Qt.Horizontal,"branch_name")
        self.model.setHeaderData(1, Qt.Horizontal,"city")
        self.model.setHeaderData(2, Qt.Horizontal, "asset")
        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.branch_name_label=QLabel("branch name")
        self.branch_name_edit=QLineEdit()

        self.city_label=QLabel("city")
        self.city_edit=QLineEdit()

        self.asset_label=QLabel("asset")
        self.assetL_edit=QLineEdit()
        self.assetL_edit.setPlaceholderText("lower")
        self.assetU_edit=QLineEdit()
        self.assetU_edit.setPlaceholderText("upper")

    
        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.branch_name_label,0,0)
        self.head_layout.addWidget(self.branch_name_edit,0,1)
        self.head_layout.addWidget(self.city_label,0,2)
        self.head_layout.addWidget(self.city_edit,0,3)
        self.head_layout.addWidget(self.asset_label,1,0)
        self.head_layout.addWidget(self.assetL_edit,1,1)
        self.head_layout.addWidget(self.assetU_edit,1,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        #self.model.setFilter("city='North' and branch_name='North_Bank'")
        self.model.select()
        
    def view_event(self):
        select_string=""
        if self.branch_name_edit.text()!="":
            select_string+="branch_name="+"'"+self.branch_name_edit.text()+"'"
        
        if self.city_edit.text()!="":
            if select_string!="":
                select_string+=" and city="+"'"+self.city_edit.text()+"'"
            else:
                select_string+="city="+"'"+self.city_edit.text()+"'"
        
        if self.assetL_edit.text()!="":
            if select_string!="":
                select_string+=" and asset >= "+self.assetL_edit.text()
            else:
                select_string+="asset >= "+self.assetL_edit.text()
        
        if self.assetU_edit.text()!="":
            if select_string!="":
                select_string+=" and asset <= "+self.assetU_edit.text()
            else:
                select_string+="asset <= "+self.assetU_edit.text()

        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("branch_name",self.branch_name_edit.text())
            record.setValue("city",self.city_edit.text())
            record.setValue("asset",self.assetL_edit.text())
            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', 'Could not Update.')
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.branch_name_edit.text())
        self.model.setData(self.model.index(row,1),self.city_edit.text())
        self.model.setData(self.model.index(row,2),self.assetL_edit.text())
        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', 'Could not Add.')
        self.model.select()
    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            self.model.removeRow(self.tableWidget.currentIndex().row())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()


class Employee_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Employee_Dialog, self).__init__(*args, **kwargs)  

class Customer_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Customer_Dialog, self).__init__(*args, **kwargs)  

class Saving_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Saving_Dialog, self).__init__(*args, **kwargs)  

class Check_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Check_Dialog, self).__init__(*args, **kwargs)  

class Loan_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Loan_Dialog, self).__init__(*args, **kwargs)  

class Statistics_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Statistics_Dialog, self).__init__(*args, **kwargs)     



class Main_Window(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Main_Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Bank Management System by drdh")
        self.setMinimumSize(800, 600)

        #define menu
        branch_menu=self.menuBar().addMenu("&branch")
        employee_menu=self.menuBar().addMenu("&employee")

        customer_menu=self.menuBar().addMenu("&customer")

        account_menu=self.menuBar().addMenu("&account")
        saving_account_menu=account_menu.addMenu("&saving")
        check_account_menu=account_menu.addMenu("&check")

        loan_menu=self.menuBar().addMenu("&loan")

        statistics_menu=self.menuBar().addMenu("&statistics")

        #define menu actions
        branch_action=QAction("branch",self)
        branch_action.triggered.connect(self.branch_dialog)
        branch_menu.addAction(branch_action)

        employee_action=QAction("employee",self)
        employee_action.triggered.connect(self.employee_dialog)
        employee_menu.addAction(employee_action)

        customer_action=QAction("customer",self)
        customer_action.triggered.connect(self.customer_dialog)
        customer_menu.addAction(customer_action)

        saving_action=QAction("saving",self)
        saving_action.triggered.connect(self.saving_dialog)
        saving_account_menu.addAction(saving_action)

        check_action=QAction("check",self)
        check_action.triggered.connect(self.check_dialog)
        check_account_menu.addAction(check_action)

        loan_action=QAction("loan",self)
        loan_action.triggered.connect(self.loan_dialog)
        loan_menu.addAction(loan_action)

        statistics_action=QAction("statistics",self)
        statistics_action.triggered.connect(self.statistics_dialog)
        statistics_menu.addAction(statistics_action)

        
    #action ==> dialog  
    def branch_dialog(self):
        dlg=Branch_Dialog()
        dlg.exec_()

    def employee_dialog(self):
        dlg=Employee_Dialog()
        dlg.exec_()

    def customer_dialog(self):
        dlg=Customer_Dialog()
        dlg.exec_()

    def saving_dialog(self):
        dlg=Saving_Dialog()
        dlg.exec_()
    
    def check_dialog(self):
        dlg=Check_Dialog()
        dlg.exec_()

    def loan_dialog(self):
        dlg=Loan_Dialog()
        dlg.exec_()

    def statistics_dialog(self):
        dlg=Statistics_Dialog()
        dlg.exec_()


class Login_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Login_Dialog, self).__init__(*args, **kwargs)

        self.setFixedWidth(300)
        self.setFixedHeight(120)

        layout = QVBoxLayout()

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
        layout.addWidget(self.passinput)
        layout.addWidget(self.QBtn)
        self.setLayout(layout)

    def login(self):
        if(self.passinput.text() == "lx"):
            self.done(2)
        elif(self.passinput.text() == "drdh"):
            #self.accept()
            self.done(2)
        else:
            QMessageBox.warning(self, 'Error', 'Wrong Password')



app = QApplication(sys.argv)
passdlg = Login_Dialog()
if(passdlg.exec_() == 2): 
    window = Main_Window()
    window.show()
#    window.loaddata()
sys.exit(app.exec_())