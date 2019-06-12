from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtSql import *
from PyQt5.QtChart import *
import sys,sqlite3,time

import os
import json

from Login_Dialog import Login_Dialog
from Branch_Dialog import Branch_Dialog


class Employee_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Employee_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Employee")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('employee')
        #self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
        #self.model.setRelation(2, QSqlRelation("employee", "employee_id", "name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "employee_id")
        self.model.setHeaderData(1, Qt.Horizontal, "branch_name")
        self.model.setHeaderData(2, Qt.Horizontal, "manager_id")
        self.model.setHeaderData(3, Qt.Horizontal, "name")
        self.model.setHeaderData(4, Qt.Horizontal, "address")
        self.model.setHeaderData(5, Qt.Horizontal, "phone")
        self.model.setHeaderData(6, Qt.Horizontal, "start_date")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.employee_id_label=QLabel("employee_id")
        self.employee_id_edit=QLineEdit()

        self.branch_name_label=QLabel("branch_name")
        self.branch_name_edit=QLineEdit()

        self.manager_id_label=QLabel("manager_id")
        self.manager_id_edit=QLineEdit()

        self.name_label=QLabel("name")
        self.name_edit=QLineEdit()

        self.address_label=QLabel("address")
        self.address_edit=QLineEdit()

        self.phone_label=QLabel("phone")
        self.phone_edit=QLineEdit()

        self.start_date_label=QLabel("start_date")
        self.start_date_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.employee_id_label,0,0)
        self.head_layout.addWidget(self.employee_id_edit,0,1)
        self.head_layout.addWidget(self.branch_name_label,0,2)
        self.head_layout.addWidget(self.branch_name_edit,0,3)
        self.head_layout.addWidget(self.manager_id_label,1,0)
        self.head_layout.addWidget(self.manager_id_edit,1,1)
        self.head_layout.addWidget(self.name_label,1,2)
        self.head_layout.addWidget(self.name_edit,1,3)
        self.head_layout.addWidget(self.address_label,2,0)
        self.head_layout.addWidget(self.address_edit,2,1)
        self.head_layout.addWidget(self.phone_label,2,2)
        self.head_layout.addWidget(self.phone_edit,2,3)
        self.head_layout.addWidget(self.start_date_label,3,0)
        self.head_layout.addWidget(self.start_date_edit,3,1)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()
    
    def view_event(self):
        select_string=""

        if self.employee_id_edit.text()!="":
            select_string+="employee_id="+"'"+self.employee_id_edit.text()+"'"
            
        if self.branch_name_edit.text()!="":
            if select_string!="":
                select_string+=" and branch_name="+"'"+self.branch_name_edit.text()+"'"
            else:
                select_string+="branch_name="+"'"+self.branch_name_edit.text()+"'"

        if self.manager_id_edit.text()!="":
            if select_string!="":
                select_string+=" and manager_id="+"'"+self.manager_id_edit.text()+"'"
            else:
                select_string+="manager_id="+"'"+self.manager_id_edit.text()+"'"
        
        if self.name_edit.text()!="":
            if select_string!="":
                select_string+=" and name="+"'"+self.name_edit.text()+"'"
            else:
                select_string+="name="+"'"+self.name_edit.text()+"'"

        if self.address_edit.text()!="":
            if select_string!="":
                select_string+=" and address="+"'"+self.address_edit.text()+"'"
            else:
                select_string+="address="+"'"+self.address_edit.text()+"'"

        if self.phone_edit.text()!="":
            if select_string!="":
                select_string+=" and phone="+"'"+self.phone_edit.text()+"'"
            else:
                select_string+="phone="+"'"+self.phone_edit.text()+"'"

        if self.start_date_edit.text()!="":
            if select_string!="":
                select_string+=" and start_date="+"'"+self.start_date_edit.text()+"'"
            else:
                select_string+="start_date="+"'"+self.start_date_edit.text()+"'"

        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("employee_id",self.employee_id_edit.text())
            record.setValue("branch_name",self.branch_name_edit.text())
            record.setValue("manager_id",self.manager_id_edit.text())
            record.setValue("name",self.name_edit.text())
            record.setValue("address",self.address_edit.text())
            record.setValue("phone",self.phone_edit.text())
            record.setValue("start_date",self.start_date_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.employee_id_edit.text())
        self.model.setData(self.model.index(row,1),self.branch_name_edit.text())
        self.model.setData(self.model.index(row,2),self.manager_id_edit.text())
        self.model.setData(self.model.index(row,3),self.name_edit.text())
        self.model.setData(self.model.index(row,4),self.address_edit.text())
        self.model.setData(self.model.index(row,5),self.phone_edit.text())
        self.model.setData(self.model.index(row,6),self.start_date_edit.text())

        if not self.model.submitAll():
            QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()




class Customer_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Customer_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Customer")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('customer')
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "customer_id")
        self.model.setHeaderData(1, Qt.Horizontal, "name")
        self.model.setHeaderData(2, Qt.Horizontal, "phone")
        self.model.setHeaderData(3, Qt.Horizontal, "address")
        self.model.setHeaderData(4, Qt.Horizontal, "contact_name")
        self.model.setHeaderData(5, Qt.Horizontal, "contact_phone")
        self.model.setHeaderData(6, Qt.Horizontal, "contact_email")
        self.model.setHeaderData(7, Qt.Horizontal, "contact_relation")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.customer_id_label=QLabel("customer_id")
        self.customer_id_edit=QLineEdit()

        self.name_label=QLabel("name")
        self.name_edit=QLineEdit()

        self.phone_label=QLabel("phone")
        self.phone_edit=QLineEdit()

        self.address_label=QLabel("address")
        self.address_edit=QLineEdit()

        self.contact_name_label=QLabel("contact_name")
        self.contact_name_edit=QLineEdit()

        self.contact_phone_label=QLabel("contact_phone")
        self.contact_phone_edit=QLineEdit()

        self.contact_email_label=QLabel("contact_email")
        self.contact_email_edit=QLineEdit()

        self.contact_relation_label=QLabel("contact_relation")
        self.contact_relation_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.customer_id_label,0,0)
        self.head_layout.addWidget(self.customer_id_edit,0,1)
        self.head_layout.addWidget(self.name_label,0,2)
        self.head_layout.addWidget(self.name_edit,0,3)
        self.head_layout.addWidget(self.phone_label,1,0)
        self.head_layout.addWidget(self.phone_edit,1,1)
        self.head_layout.addWidget(self.address_label,1,2)
        self.head_layout.addWidget(self.address_edit,1,3)
        self.head_layout.addWidget(self.contact_name_label,2,0)
        self.head_layout.addWidget(self.contact_name_edit,2,1)
        self.head_layout.addWidget(self.contact_phone_label,2,2)
        self.head_layout.addWidget(self.contact_phone_edit,2,3)
        self.head_layout.addWidget(self.contact_email_label,3,0)
        self.head_layout.addWidget(self.contact_email_edit,3,1)
        self.head_layout.addWidget(self.contact_relation_label,3,2)
        self.head_layout.addWidget(self.contact_relation_edit,3,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(900, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.customer_id_edit.text()!="":
            select_string+="customer_id="+"'"+self.customer_id_edit.text()+"'"

        if self.name_edit.text()!="":
            if select_string!="":
                select_string+=" and name="+"'"+self.name_edit.text()+"'"
            else:
                select_string+="name="+"'"+self.name_edit.text()+"'"

        if self.phone_edit.text()!="":
            if select_string!="":
                select_string+=" and phone="+"'"+self.phone_edit.text()+"'"
            else:
                select_string+="phone="+"'"+self.phone_edit.text()+"'"

        if self.address_edit.text()!="":
            if select_string!="":
                select_string+=" and address="+"'"+self.address_edit.text()+"'"
            else:
                select_string+="address="+"'"+self.address_edit.text()+"'"

        if self.contact_name_edit.text()!="":
            if select_string!="":
                select_string+=" and contact_name="+"'"+self.contact_name_edit.text()+"'"
            else:
                select_string+="contact_name="+"'"+self.contact_name_edit.text()+"'"

        if self.contact_phone_edit.text()!="":
            if select_string!="":
                select_string+=" and contact_phone="+"'"+self.contact_phone_edit.text()+"'"
            else:
                select_string+="contact_phone="+"'"+self.contact_phone_edit.text()+"'"

        if self.contact_email_edit.text()!="":
            if select_string!="":
                select_string+=" and contact_email="+"'"+self.contact_email_edit.text()+"'"
            else:
                select_string+="contact_email="+"'"+self.contact_email_edit.text()+"'"

        if self.contact_relation_edit.text()!="":
            if select_string!="":
                select_string+=" and contact_relation="+"'"+self.contact_relation_edit.text()+"'"
            else:
                select_string+="contact_relation="+"'"+self.contact_relation_edit.text()+"'"
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("customer_id",self.customer_id_edit.text())
            record.setValue("name",self.name_edit.text())
            record.setValue("phone",self.phone_edit.text())
            record.setValue("address",self.address_edit.text())
            record.setValue("contact_name",self.contact_name_edit.text())
            record.setValue("contact_phone",self.contact_phone_edit.text())
            record.setValue("contact_email",self.contact_email_edit.text())
            record.setValue("contact_relation",self.contact_relation_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.customer_id_edit.text())
        self.model.setData(self.model.index(row,1),self.name_edit.text())
        self.model.setData(self.model.index(row,2),self.phone_edit.text())
        self.model.setData(self.model.index(row,3),self.address_edit.text())
        self.model.setData(self.model.index(row,4),self.contact_name_edit.text())
        self.model.setData(self.model.index(row,5),self.contact_phone_edit.text())
        self.model.setData(self.model.index(row,6),self.contact_email_edit.text())
        self.model.setData(self.model.index(row,7),self.contact_relation_edit.text())

        if not self.model.submitAll():
            QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Responsible_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Responsible_Dialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Responsible")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('responsible')
        #self.model.setRelation(0, QSqlRelation("employee", "employee_id", "name"))
        #self.model.setRelation(1, QSqlRelation("customer", "customer_id", "name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "employee_id")
        self.model.setHeaderData(1, Qt.Horizontal, "customer_id")
        self.model.setHeaderData(2, Qt.Horizontal, "type")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.employee_id_label=QLabel("employee_id")
        self.employee_id_edit=QLineEdit()

        self.customer_id_label=QLabel("customer_id")
        self.customer_id_edit=QLineEdit()

        self.type_label=QLabel("type")
        self.type_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.employee_id_label,0,0)
        self.head_layout.addWidget(self.employee_id_edit,0,1)
        self.head_layout.addWidget(self.customer_id_label,0,2)
        self.head_layout.addWidget(self.customer_id_edit,0,3)
        self.head_layout.addWidget(self.type_label,1,0)
        self.head_layout.addWidget(self.type_edit,1,1)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.employee_id_edit.text()!="":
            select_string+="employee_id="+"'"+self.employee_id_edit.text()+"'"

        if self.customer_id_edit.text()!="":
            if select_string!="":
                select_string+=" and customer_id="+"'"+self.customer_id_edit.text()+"'"
            else:
                select_string+="customer_id="+"'"+self.customer_id_edit.text()+"'"

        if self.type_edit.text()!="":
            if select_string!="":
                select_string+=" and type="+"'"+self.type_edit.text()+"'"
            else:
                select_string+="type="+"'"+self.type_edit.text()+"'"
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("employee_id",self.employee_id_edit.text())
            record.setValue("customer_id",self.customer_id_edit.text())
            record.setValue("type",self.type_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.employee_id_edit.text())
        self.model.setData(self.model.index(row,1),self.customer_id_edit.text())
        self.model.setData(self.model.index(row,2),self.type_edit.text())

        if not self.model.submitAll():
            QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()


    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Saving_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Saving_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Saving Account")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('saving_account')
        #self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "account_id")
        self.model.setHeaderData(1, Qt.Horizontal, "branch_name")
        self.model.setHeaderData(2, Qt.Horizontal, "balance")
        self.model.setHeaderData(3, Qt.Horizontal, "open_date")
        self.model.setHeaderData(4, Qt.Horizontal, "interest_rate")
        self.model.setHeaderData(5, Qt.Horizontal, "currency_type")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.account_id_label=QLabel("account_id")
        self.account_id_edit=QLineEdit()

        self.branch_name_label=QLabel("branch_name")
        self.branch_name_edit=QLineEdit()

        self.balance_label=QLabel("balance")
        self.balanceL_edit=QLineEdit()
        self.balanceL_edit.setPlaceholderText("lower")
        self.balanceU_edit=QLineEdit()
        self.balanceU_edit.setPlaceholderText("upper")

        self.open_date_label=QLabel("open_date")
        self.open_date_edit=QLineEdit()

        self.interest_rate_label=QLabel("interest_rate")
        self.interest_rateL_edit=QLineEdit()
        self.interest_rateL_edit.setPlaceholderText("lower")
        self.interest_rateU_edit=QLineEdit()
        self.interest_rateU_edit.setPlaceholderText("upper")

        self.currency_type_label=QLabel("currency_type")
        self.currency_type_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.account_id_label,0,0)
        self.head_layout.addWidget(self.account_id_edit,0,1)
        self.head_layout.addWidget(self.branch_name_label,0,2)
        self.head_layout.addWidget(self.branch_name_edit,0,3)
        self.head_layout.addWidget(self.balance_label,1,0)
        self.head_layout.addWidget(self.balanceL_edit,1,1)
        self.head_layout.addWidget(self.balanceU_edit,1,3)
        self.head_layout.addWidget(self.open_date_label,2,0)
        self.head_layout.addWidget(self.open_date_edit,2,1)
        self.head_layout.addWidget(self.interest_rate_label,3,0)
        self.head_layout.addWidget(self.interest_rateL_edit,3,1)
        self.head_layout.addWidget(self.interest_rateU_edit,3,3)
        self.head_layout.addWidget(self.currency_type_label,2,2)
        self.head_layout.addWidget(self.currency_type_edit,2,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.account_id_edit.text()!="":
            select_string+="account_id="+"'"+self.account_id_edit.text()+"'"

        if self.branch_name_edit.text()!="":
            if select_string!="":
                select_string+=" and branch_name="+"'"+self.branch_name_edit.text()+"'"
            else:
                select_string+="branch_name="+"'"+self.branch_name_edit.text()+"'"

        if self.balanceL_edit.text()!="":
            if select_string!="":
                select_string+=" and balance >= "+self.balanceL_edit.text()
            else:
                select_string+="balance >= "+self.balanceL_edit.text()
        
        if self.balanceU_edit.text()!="":
            if select_string!="":
                select_string+=" and balance <= "+self.balanceU_edit.text()
            else:
                select_string+="balance <= "+self.balanceU_edit.text()

        if self.open_date_edit.text()!="":
            if select_string!="":
                select_string+=" and open_date="+"'"+self.open_date_edit.text()+"'"
            else:
                select_string+="open_date="+"'"+self.open_date_edit.text()+"'"

        if self.interest_rateL_edit.text()!="":
            if select_string!="":
                select_string+=" and interest_rate >= "+self.interest_rateL_edit.text()
            else:
                select_string+="interest_rate >= "+self.interest_rateL_edit.text()
        
        if self.interest_rateU_edit.text()!="":
            if select_string!="":
                select_string+=" and interest_rate <= "+self.interest_rateU_edit.text()
            else:
                select_string+="interest_rate <= "+self.interest_rateU_edit.text()
        
        if self.currency_type_edit.text()!="":
            if select_string!="":
                select_string+=" and currency_type="+"'"+self.currency_type_edit.text()+"'"
            else:
                select_string+="currency_type="+"'"+self.currency_type_edit.text()+"'"
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("account_id",self.account_id_edit.text())
            record.setValue("branch_name",self.branch_name_edit.text())
            record.setValue("balance",self.balanceL_edit.text())
            record.setValue("open_date",self.open_date_edit.text())
            record.setValue("interest_rate",self.interest_rateL_edit.text())
            record.setValue("currency_type",self.currency_type_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.account_id_edit.text())
        self.model.setData(self.model.index(row,1),self.branch_name_edit.text())
        self.model.setData(self.model.index(row,2),self.balanceL_edit.text())
        self.model.setData(self.model.index(row,3),self.open_date_edit.text())
        self.model.setData(self.model.index(row,4),self.interest_rateL_edit.text())
        self.model.setData(self.model.index(row,5),self.currency_type_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Check_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Check_Dialog, self).__init__(*args, **kwargs)
        self.setWindowTitle("Check Account") 
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('check_account')
        #self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "account_id")
        self.model.setHeaderData(1, Qt.Horizontal, "branch_name")
        self.model.setHeaderData(2, Qt.Horizontal, "balance")
        self.model.setHeaderData(3, Qt.Horizontal, "open_date")
        self.model.setHeaderData(4, Qt.Horizontal, "overdraft")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.account_id_label=QLabel("account_id")
        self.account_id_edit=QLineEdit()

        self.branch_name_label=QLabel("branch_name")
        self.branch_name_edit=QLineEdit()

        self.balance_label=QLabel("balance")
        self.balanceL_edit=QLineEdit()
        self.balanceL_edit.setPlaceholderText("lower")
        self.balanceU_edit=QLineEdit()
        self.balanceU_edit.setPlaceholderText("upper")

        self.open_date_label=QLabel("open_date")
        self.open_date_edit=QLineEdit()

        self.overdraft_label=QLabel("overdraft")
        self.overdraftL_edit=QLineEdit()
        self.overdraftL_edit.setPlaceholderText("lower")
        self.overdraftU_edit=QLineEdit()
        self.overdraftU_edit.setPlaceholderText("upper")

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.account_id_label,0,0)
        self.head_layout.addWidget(self.account_id_edit,0,1)
        self.head_layout.addWidget(self.branch_name_label,0,2)
        self.head_layout.addWidget(self.branch_name_edit,0,3)
        self.head_layout.addWidget(self.balance_label,1,0)
        self.head_layout.addWidget(self.balanceL_edit,1,1)
        self.head_layout.addWidget(self.balanceU_edit,1,3)
        self.head_layout.addWidget(self.open_date_label,2,0)
        self.head_layout.addWidget(self.open_date_edit,2,1)
        self.head_layout.addWidget(self.overdraft_label,3,0)
        self.head_layout.addWidget(self.overdraftL_edit,3,1)
        self.head_layout.addWidget(self.overdraftU_edit,3,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.account_id_edit.text()!="":
            select_string+="account_id="+"'"+self.account_id_edit.text()+"'"

        if self.branch_name_edit.text()!="":
            if select_string!="":
                select_string+=" and branch_name="+"'"+self.branch_name_edit.text()+"'"
            else:
                select_string+="branch_name="+"'"+self.branch_name_edit.text()+"'"

        if self.balanceL_edit.text()!="":
            if select_string!="":
                select_string+=" and balance >= "+self.balanceL_edit.text()
            else:
                select_string+="balance >= "+self.balanceL_edit.text()
        
        if self.balanceU_edit.text()!="":
            if select_string!="":
                select_string+=" and balance <= "+self.balanceU_edit.text()
            else:
                select_string+="balance <= "+self.balanceU_edit.text()
        
        if self.open_date_edit.text()!="":
            if select_string!="":
                select_string+=" and open_date="+"'"+self.open_date_edit.text()+"'"
            else:
                select_string+="open_date="+"'"+self.open_date_edit.text()+"'"

        if self.overdraftL_edit.text()!="":
            if select_string!="":
                select_string+=" and overdraft >= "+self.overdraftL_edit.text()
            else:
                select_string+="overdraft >= "+self.overdraftL_edit.text()
        
        if self.overdraftU_edit.text()!="":
            if select_string!="":
                select_string+=" and overdraft <= "+self.overdraftU_edit.text()
            else:
                select_string+="overdraft <= "+self.overdraftU_edit.text()
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("account_id",self.account_id_edit.text())
            record.setValue("branch_name",self.branch_name_edit.text())
            record.setValue("balance",self.balanceL_edit.text())
            record.setValue("open_date",self.open_date_edit.text())
            record.setValue("overdraft",self.overdraftL_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.account_id_edit.text())
        self.model.setData(self.model.index(row,1),self.branch_name_edit.text())
        self.model.setData(self.model.index(row,2),self.balanceL_edit.text())
        self.model.setData(self.model.index(row,3),self.open_date_edit.text())
        self.model.setData(self.model.index(row,4),self.overdraftL_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Depositor_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Depositor_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Depositor") 
        #ui
        self.tableWidget = QTableView()
        #self.view= QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('depositor')
        #self.model.setRelation(0, QSqlRelation("account", "account_id", "account_id"))
        #self.model.setRelation(1, QSqlRelation("customer", "customer_id", "name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "account")
        self.model.setHeaderData(1, Qt.Horizontal, "customer")
        self.model.setHeaderData(2, Qt.Horizontal, "access_date")

        self.tableWidget.setModel(self.model)
        #self.view.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.account_id_label=QLabel("account_id")
        self.account_id_edit=QLineEdit()

        self.customer_id_label=QLabel("customer_id")
        self.customer_id_edit=QLineEdit()

        self.access_date_label=QLabel("access_date")
        self.access_date_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.account_id_label,0,0)
        self.head_layout.addWidget(self.account_id_edit,0,1)
        self.head_layout.addWidget(self.customer_id_label,0,2)
        self.head_layout.addWidget(self.customer_id_edit,0,3)
        self.head_layout.addWidget(self.access_date_label,1,0)
        self.head_layout.addWidget(self.access_date_edit,1,1)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        #self.tableLayout=QHBoxLayout()
        #self.tableLayout.addWidget(self.tableWidget)
        #self.tableLayout.addWidget(self.view)
        #self.layout.addLayout(self.tableLayout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.account_id_edit.text()!="":
            select_string+="account_id="+"'"+self.account_id_edit.text()+"'"
            #select_string+="account_0="+"'"+self.account_id_edit.text()+"'"

        if self.customer_id_edit.text()!="":
            if select_string!="":
                select_string+=" and customer_id="+"'"+self.customer_id_edit.text()+"'"
            else:
                select_string+="customer_id="+"'"+self.customer_id_edit.text()+"'"

        if self.access_date_edit.text()!="":
            if select_string!="":
                select_string+=" and access_date="+"'"+self.access_date_edit.text()+"'"
            else:
                select_string+="access_date="+"'"+self.access_date_edit.text()+"'"

        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("account_id",self.account_id_edit.text())
            record.setValue("customer_id",self.customer_id_edit.text())
            record.setValue("access_date",self.access_date_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.account_id_edit.text())
        self.model.setData(self.model.index(row,1),self.customer_id_edit.text())
        self.model.setData(self.model.index(row,2),self.access_date_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Loan_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Loan_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Loan") 
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('loan')
        #self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "loan_id")
        self.model.setHeaderData(1, Qt.Horizontal, "branch_name")
        self.model.setHeaderData(2, Qt.Horizontal, "amount")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.loan_id_label=QLabel("loan_id")
        self.loan_id_edit=QLineEdit()

        self.branch_name_label=QLabel("branch_name")
        self.branch_name_edit=QLineEdit()

        self.amount_label=QLabel("amount")
        self.amountL_edit=QLineEdit()
        self.amountL_edit.setPlaceholderText("lower")
        self.amountU_edit=QLineEdit()
        self.amountU_edit.setPlaceholderText("upper")

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.loan_id_label,0,0)
        self.head_layout.addWidget(self.loan_id_edit,0,1)
        self.head_layout.addWidget(self.branch_name_label,0,2)
        self.head_layout.addWidget(self.branch_name_edit,0,3)
        self.head_layout.addWidget(self.amount_label,2,0)
        self.head_layout.addWidget(self.amountL_edit,2,1)
        self.head_layout.addWidget(self.amountU_edit,2,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.loan_id_edit.text()!="":
            select_string+="loan_id="+"'"+self.loan_id_edit.text()+"'"

        if self.branch_name_edit.text()!="":
            if select_string!="":
                select_string+=" and branch_name="+"'"+self.branch_name_edit.text()+"'"
            else:
                select_string+="branch_name="+"'"+self.branch_name_edit.text()+"'"
        
        if self.amountL_edit.text()!="":
            if select_string!="":
                select_string+=" and amount >= "+self.amountL_edit.text()
            else:
                select_string+="amount >= "+self.amountL_edit.text()
        
        if self.amountU_edit.text()!="":
            if select_string!="":
                select_string+=" and amount <= "+self.amountU_edit.text()
            else:
                select_string+="amount <= "+self.amountU_edit.text()
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("loan_id",self.loan_id_edit.text())
            record.setValue("branch_name",self.branch_name_edit.text())
            record.setValue("amount",self.amountL_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.loan_id_edit.text())
        self.model.setData(self.model.index(row,1),self.branch_name_edit.text())
        self.model.setData(self.model.index(row,2),self.amountL_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Payment_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Payment_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Payment")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('payment')
        #self.model.setRelation(0, QSqlRelation("loan", "loan_id", "loan_id"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "loan_id")
        self.model.setHeaderData(1, Qt.Horizontal, "payment_id")
        self.model.setHeaderData(2, Qt.Horizontal, "payment_date")
        self.model.setHeaderData(3, Qt.Horizontal, "amount")
        self.model.setHeaderData(4, Qt.Horizontal, "total")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.loan_id_label=QLabel("loan_id")
        self.loan_id_edit=QLineEdit()

        self.payment_id_label=QLabel("payment_id")
        self.payment_id_edit=QLineEdit()

        self.payment_date_label=QLabel("payment_date")
        self.payment_date_edit=QLineEdit()

        self.amount_label=QLabel("amount")
        self.amountL_edit=QLineEdit()
        self.amountL_edit.setPlaceholderText("lower")
        self.amountU_edit=QLineEdit()
        self.amountU_edit.setPlaceholderText("upper")

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.loan_id_label,0,0)
        self.head_layout.addWidget(self.loan_id_edit,0,1)
        self.head_layout.addWidget(self.payment_id_label,0,2)
        self.head_layout.addWidget(self.payment_id_edit,0,3)
        self.head_layout.addWidget(self.payment_date_label,1,0)
        self.head_layout.addWidget(self.payment_date_edit,1,1)
        self.head_layout.addWidget(self.amount_label,2,0)
        self.head_layout.addWidget(self.amountL_edit,2,1)
        self.head_layout.addWidget(self.amountU_edit,2,3)

        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.loan_id_edit.text()!="":
            select_string+="loan_id="+"'"+self.loan_id_edit.text()+"'"

        if self.payment_id_edit.text()!="":
            if select_string!="":
                select_string+=" and payment_id="+"'"+self.payment_id_edit.text()+"'"
            else:
                select_string+="payment_id="+"'"+self.payment_id_edit.text()+"'"

        if self.payment_date_edit.text()!="":
            if select_string!="":
                select_string+=" and payment_date="+"'"+self.payment_date_edit.text()+"'"
            else:
                select_string+="payment_date="+"'"+self.payment_date_edit.text()+"'"
        
        if self.amountL_edit.text()!="":
            if select_string!="":
                select_string+=" and amount >= "+self.amountL_edit.text()
            else:
                select_string+="amount >= "+self.amountL_edit.text()
        
        if self.amountU_edit.text()!="":
            if select_string!="":
                select_string+=" and amount <= "+self.amountU_edit.text()
            else:
                select_string+="amount <= "+self.amountU_edit.text()
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("loan_id",self.loan_id_edit.text())
            record.setValue("payment_id",self.payment_id_edit.text())
            record.setValue("payment_date",self.payment_date_edit.text())
            record.setValue("amount",self.amountL_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.loan_id_edit.text())
        self.model.setData(self.model.index(row,1),self.payment_id_edit.text())
        self.model.setData(self.model.index(row,2),self.payment_date_edit.text())
        self.model.setData(self.model.index(row,3),self.amountL_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Borrow_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Borrow_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Borrow")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('borrow')
        #self.model.setRelation(0, QSqlRelation("customer", "customer_id", "name"))
        #self.model.setRelation(1, QSqlRelation("loan", "loan_id", "loan_id"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "customer_id")
        self.model.setHeaderData(1, Qt.Horizontal, "loan_id")

        self.tableWidget.setModel(self.model)
        self.tableWidget.setItemDelegate(QSqlRelationalDelegate(self.tableWidget))

        self.customer_id_label=QLabel("customer_id")
        self.customer_id_edit=QLineEdit()

        self.loan_id_label=QLabel("loan_id")
        self.loan_id_edit=QLineEdit()

        self.view_button=QPushButton("view")
        self.view_button.clicked.connect(self.view_event)
        self.update_button=QPushButton("update")
        self.update_button.clicked.connect(self.update_event)
        self.add_button=QPushButton("add")
        self.add_button.clicked.connect(self.add_event)
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
        self.delete_button=QPushButton("delete")
        self.delete_button.clicked.connect(self.delete_event)

        self.head_layout=QGridLayout()
        self.head_layout.addWidget(self.customer_id_label,0,0)
        self.head_layout.addWidget(self.customer_id_edit,0,1)
        self.head_layout.addWidget(self.loan_id_label,0,2)
        self.head_layout.addWidget(self.loan_id_edit,0,3)
        
        self.button_layout=QHBoxLayout()
        self.button_layout.addWidget(self.view_button)
        self.button_layout.addWidget(self.update_button)
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.new_button)
        self.button_layout.addWidget(self.delete_button)

        self.layout=QVBoxLayout()
        self.layout.addLayout(self.head_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)
        self.setMinimumSize(800, 600)

        self.model.select()

    def view_event(self):
        select_string=""

        if self.customer_id_edit.text()!="":
            select_string+="customer_id="+"'"+self.customer_id_edit.text()+"'"

        if self.loan_id_edit.text()!="":
            if select_string!="":
                select_string+=" and loan_id="+"'"+self.loan_id_edit.text()+"'"
            else:
                select_string+="loan_id="+"'"+self.loan_id_edit.text()+"'"
        
        self.model.setFilter(select_string)
        self.model.select()

    def update_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            record=self.model.record(self.tableWidget.currentIndex().row())
            record.setValue("customer_id",self.customer_id_edit.text())
            record.setValue("loan_id",self.loan_id_edit.text())

            if not self.model.setRecord(self.tableWidget.currentIndex().row(),record):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to update.", QMessageBox.Ok)
            self.show()
        self.model.select()

    def add_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.setData(self.model.index(row,0),self.customer_id_edit.text())
        self.model.setData(self.model.index(row,1),self.loan_id_edit.text())

        if not self.model.submitAll():
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
        self.model.select()
    
    def new_event(self):
        row=self.model.rowCount()
        self.model.insertRows(row,1)
        self.model.submitAll()

    
    def delete_event(self):
        if self.tableWidget.currentIndex().row() > -1:
            if not self.model.removeRow(self.tableWidget.currentIndex().row()):
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
            self.model.select()
        else:
            QMessageBox.question(self,'Message', "Please select a row would you like to delete", QMessageBox.Ok)
            self.show()

class Statistics_Overall_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Statistics_Overall_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Statistics Overall") 
        self.setMinimumSize(800, 800)
        #ui 
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db") 
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")
        
        #sum(loan) for each branch
        query_loan=QSqlQuery()
        query_loan.exec("select branch_name,sum(amount) from loan group by branch_name;")

        result_loan=[]
        while query_loan.next():# branch_name,sum(amount)
            result_loan.append((query_loan.value(0),query_loan.value(1)))
        
        series_loan=QPieSeries()
        for r in result_loan:
            series_loan.append(r[0],r[1])
        series_loan.setLabelsVisible()

        chart_loan=QChart()
        chart_loan.addSeries(series_loan)
        chart_loan.setTitle("Toal Loan for every Branch")
        chart_loan.setAnimationOptions(QChart.AllAnimations)
        chart_loan.legend().hide()

        chart_view_loan=QChartView(chart_loan)
        chart_view_loan.setRenderHint(QPainter.Antialiasing)

        #sum(saving_account)
        query_saving=QSqlQuery()
        query_saving.exec("select branch_name,sum(balance) from saving_account group by branch_name;")

        result_saving=[]
        while query_saving.next():# branch_name,sum(amount)
            result_saving.append((query_saving.value(0),query_saving.value(1)))
        
        series_saving=QPieSeries()
        for r in result_saving:
            series_saving.append(r[0],r[1])
        series_saving.setLabelsVisible()

        chart_saving=QChart()
        chart_saving.addSeries(series_saving)
        chart_saving.setTitle("Saving Account for every Branch")
        chart_saving.setAnimationOptions(QChart.AllAnimations)
        chart_saving.legend().hide()

        chart_view_saving=QChartView(chart_saving)
        chart_view_saving.setRenderHint(QPainter.Antialiasing)

        #sum(check_account)
        query_check=QSqlQuery()
        query_check.exec("select branch_name,sum(balance) from saving_account group by branch_name;")

        result_check=[]
        while query_check.next():# branch_name,sum(amount)
            result_check.append((query_check.value(0),query_check.value(1)))
        
        series_check=QPieSeries()
        for r in result_check:
            series_check.append(r[0],r[1])
        series_check.setLabelsVisible()

        chart_check=QChart()
        chart_check.addSeries(series_check)
        chart_check.setTitle("Check Account for every Branch")
        chart_check.setAnimationOptions(QChart.AllAnimations)
        chart_check.legend().hide()

        chart_view_check=QChartView(chart_check)
        chart_view_check.setRenderHint(QPainter.Antialiasing)

        #bank_asset
        query_branchAsset=QSqlQuery()
        query_branchAsset.exec("select branch_name,asset from branch;")

        result_branchAsset=[]
        while query_branchAsset.next():# branch_name,sum(amount)
            result_branchAsset.append((query_branchAsset.value(0),query_branchAsset.value(1)))
        
        series_branchAsset=QPieSeries()
        for r in result_branchAsset:
            series_branchAsset.append(r[0],r[1])
        series_branchAsset.setLabelsVisible()

        chart_branchAsset=QChart()
        chart_branchAsset.addSeries(series_branchAsset)
        chart_branchAsset.setTitle("Branch Asset Account for every Branch")
        chart_branchAsset.setAnimationOptions(QChart.AllAnimations)
        chart_branchAsset.legend().hide()

        chart_view_branchAsset=QChartView(chart_branchAsset)
        chart_view_branchAsset.setRenderHint(QPainter.Antialiasing)

    
        self.layout=QGridLayout()
        self.layout.addWidget(chart_view_loan,0,0)
        self.layout.addWidget(chart_view_saving,0,1)
        self.layout.addWidget(chart_view_check,1,0)
        self.layout.addWidget(chart_view_branchAsset,1,1)
        self.setLayout(self.layout)
        
        #slice_loan=series_loan.slices()[1]
        #slice_loan.setExploded()
        #slice_loan.setLabelVisible()
        #slice_loan.setPen(QPen(Qt.darkGreen,2))
        #slice_loan.setBrush(Qt.green)

class Statistics_Seasonly_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Statistics_Seasonly_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Statistics Seasonly") 
        self.setMinimumSize(1000, 400)
        #ui 
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db") 
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")
        
        #saving seasonly
        query_seasonly_saving=[QSqlQuery() for _ in range(4)]
        result_seasonly_saving=[]

        for i,s in enumerate([("01","03"),("04","06"),("07","09"),("10","12")]):
            query_seasonly_saving[i].exec("select branch_name,sum(balance) from saving_account where open_date>='2019-%s' and open_date<='2019-%s' group by branch_name;"%(s[0],s[1]))
            temp_result=[]
            while query_seasonly_saving[i].next():
                temp_result.append((query_seasonly_saving[i].value(0),query_seasonly_saving[i].value(1)))
            result_seasonly_saving.append(temp_result)

        set_seasonly_saving=[QBarSet(bank[0]) for bank in result_seasonly_saving[0]]
        for i in range(4):
            for j in range(len(set_seasonly_saving)):
                set_seasonly_saving[j].append(result_seasonly_saving[i][j][1])
        
        series_seasonly_saving=QBarSeries()
        for s in set_seasonly_saving:
            series_seasonly_saving.append(s)
        
        chart_seasonly_saving=QChart()
        chart_seasonly_saving.addSeries(series_seasonly_saving)
        chart_seasonly_saving.setTitle("Saving Seasonly for every Branch")
        chart_seasonly_saving.setAnimationOptions(QChart.SeriesAnimations)

        categories_seasonly_saving=[]
        for i in range(4):
            categories_seasonly_saving.append(str(i+1))
        
        axisX_seasonly_saving=QBarCategoryAxis()
        axisX_seasonly_saving.append(categories_seasonly_saving)
        chart_seasonly_saving.addAxis(axisX_seasonly_saving,Qt.AlignBottom)
        series_seasonly_saving.attachAxis(axisX_seasonly_saving)

        axisY_seasonly_saving=QValueAxis()
        chart_seasonly_saving.addAxis(axisY_seasonly_saving,Qt.AlignLeft)
        series_seasonly_saving.attachAxis(axisY_seasonly_saving)

        chart_seasonly_saving.legend().setVisible(True)
        chart_seasonly_saving.legend().setAlignment(Qt.AlignBottom)

        chart_view_seasonly_saving=QChartView(chart_seasonly_saving)
        chart_view_seasonly_saving.setRenderHint(QPainter.Antialiasing)

        #check seasonly
        query_seasonly_check=[QSqlQuery() for _ in range(4)]
        result_seasonly_check=[]

        for i,s in enumerate([("01","03"),("04","06"),("07","09"),("10","12")]):
            query_seasonly_check[i].exec("select branch_name,sum(balance) from check_account where open_date>='2019-%s' and open_date<='2019-%s' group by branch_name;"%(s[0],s[1]))
            temp_result=[]
            while query_seasonly_check[i].next():
                temp_result.append((query_seasonly_check[i].value(0),query_seasonly_check[i].value(1)))
            result_seasonly_check.append(temp_result)

        set_seasonly_check=[QBarSet(bank[0]) for bank in result_seasonly_check[0]]
        for i in range(4):
            for j in range(len(set_seasonly_check)):
                set_seasonly_check[j].append(result_seasonly_check[i][j][1])
        
        series_seasonly_check=QBarSeries()
        for s in set_seasonly_check:
            series_seasonly_check.append(s)
        
        chart_seasonly_check=QChart()
        chart_seasonly_check.addSeries(series_seasonly_check)
        chart_seasonly_check.setTitle("Check Seasonly for every Branch")
        chart_seasonly_check.setAnimationOptions(QChart.SeriesAnimations)

        categories_seasonly_check=[]
        for i in range(4):
            categories_seasonly_check.append(str(i+1))
        
        axisX_seasonly_check=QBarCategoryAxis()
        axisX_seasonly_check.append(categories_seasonly_check)
        chart_seasonly_check.addAxis(axisX_seasonly_check,Qt.AlignBottom)
        series_seasonly_check.attachAxis(axisX_seasonly_check)

        axisY_seasonly_check=QValueAxis()
        chart_seasonly_check.addAxis(axisY_seasonly_check,Qt.AlignLeft)
        series_seasonly_check.attachAxis(axisY_seasonly_check)

        chart_seasonly_check.legend().setVisible(True)
        chart_seasonly_check.legend().setAlignment(Qt.AlignBottom)

        chart_view_seasonly_check=QChartView(chart_seasonly_check)
        chart_view_seasonly_check.setRenderHint(QPainter.Antialiasing)

        self.layout=QGridLayout()
        self.layout.addWidget(chart_view_seasonly_saving,0,0)
        self.layout.addWidget(chart_view_seasonly_check,0,1)
        self.setLayout(self.layout)

class Statistics_Monthly_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Statistics_Monthly_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Statistics Monthly") 
        self.setMinimumSize(1000, 800)
        #ui 
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db") 
        self.db.open()
        self.db.exec("PRAGMA foreign_keys=ON;")

        month=["Jan.","Feb.","Mar.","Apr.","May.","Jun.","Jul.","Aug.","Sep.","Oct.","Nov.","Dec."]
        #payment monthly
        query_monthly_payment=[QSqlQuery() for _ in range(12)]
        result_monthly_payment=[]

        for i,s in enumerate(["01","02","03","04","05","06","07","08","09","10","11","12"]):
            query_monthly_payment[i].exec("select branch_name,sum(payment.amount) from payment,loan \
            where payment.loan_id=loan.loan_id and payment_date like '2019-{}%' group by branch_name;".format(s))
            temp_result=[]
            while query_monthly_payment[i].next():
                temp_result.append((query_monthly_payment[i].value(0),query_monthly_payment[i].value(1)))
            result_monthly_payment.append(temp_result)

        set_monthly_payment=[QBarSet(bank[0]) for bank in result_monthly_payment[0]]
        for i in range(12):
            for j in range(len(set_monthly_payment)):
                set_monthly_payment[j].append(result_monthly_payment[i][j][1])
        
        #series_monthly_payment=QBarSeries()
        series_monthly_payment=QStackedBarSeries()
        for s in set_monthly_payment:
            series_monthly_payment.append(s)
        
        chart_monthly_payment=QChart()
        chart_monthly_payment.addSeries(series_monthly_payment)
        chart_monthly_payment.setTitle("Payment Monthly for every Branch")
        chart_monthly_payment.setAnimationOptions(QChart.SeriesAnimations)

        categories_monthly_payment=[]
        for i in month:
            categories_monthly_payment.append(i)
        
        axisX_monthly_payment=QBarCategoryAxis()
        axisX_monthly_payment.append(categories_monthly_payment)
        chart_monthly_payment.addAxis(axisX_monthly_payment,Qt.AlignBottom)
        series_monthly_payment.attachAxis(axisX_monthly_payment)

        axisY_monthly_payment=QValueAxis()
        chart_monthly_payment.addAxis(axisY_monthly_payment,Qt.AlignLeft)
        series_monthly_payment.attachAxis(axisY_monthly_payment)

        chart_monthly_payment.legend().setVisible(True)
        chart_monthly_payment.legend().setAlignment(Qt.AlignBottom)

        chart_view_monthly_payment=QChartView(chart_monthly_payment)
        chart_view_monthly_payment.setRenderHint(QPainter.Antialiasing)

        #account monthly
        query_monthly_account=[QSqlQuery() for _ in range(12)]
        result_monthly_account=[]

        for i,s in enumerate(["01","02","03","04","05","06","07","08","09","10","11","12"]):
            query_monthly_account[i].exec("select branch_name,count(*) from account,depositor \
            where account.account_id=depositor.account_id and access_date like '2019-{}%' group by branch_name;".format(s))
            temp_result=[]
            while query_monthly_account[i].next():
                temp_result.append((query_monthly_account[i].value(0),query_monthly_account[i].value(1)))
            result_monthly_account.append(temp_result)

        set_monthly_account=[QBarSet(bank[0]) for bank in result_monthly_account[0]]
        for i in range(12):
            for j in range(len(set_monthly_account)):
                set_monthly_account[j].append(result_monthly_account[i][j][1])
        
        #series_monthly_account=QBarSeries()
        series_monthly_account=QStackedBarSeries()
        for s in set_monthly_account:
            series_monthly_account.append(s)
        
        chart_monthly_account=QChart()
        chart_monthly_account.addSeries(series_monthly_account)
        chart_monthly_account.setTitle("Account Monthly for every Branch")
        chart_monthly_account.setAnimationOptions(QChart.SeriesAnimations)

        categories_monthly_account=[]
        for i in month:
            categories_monthly_account.append(i)
        
        axisX_monthly_account=QBarCategoryAxis()
        axisX_monthly_account.append(categories_monthly_account)
        chart_monthly_account.addAxis(axisX_monthly_account,Qt.AlignBottom)
        series_monthly_account.attachAxis(axisX_monthly_account)

        axisY_monthly_account=QValueAxis()
        chart_monthly_account.addAxis(axisY_monthly_account,Qt.AlignLeft)
        series_monthly_account.attachAxis(axisY_monthly_account)

        chart_monthly_account.legend().setVisible(True)
        chart_monthly_account.legend().setAlignment(Qt.AlignBottom)

        chart_view_monthly_account=QChartView(chart_monthly_account)
        chart_view_monthly_account.setRenderHint(QPainter.Antialiasing)

        self.layout=QGridLayout()
        self.layout.addWidget(chart_view_monthly_payment,0,0)
        self.layout.addWidget(chart_view_monthly_account,1,0)
        self.setLayout(self.layout)

class Main_Window(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Main_Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Bank Management System by drdh")
        #self.setMinimumSize(800, 400)
        self.setFixedWidth(800)
        self.setFixedHeight(494)

        #define menu
        infrastracture_menu=self.menuBar().addMenu("&infrastructure")
        client_menu=self.menuBar().addMenu("&client")
        transaction_menu=self.menuBar().addMenu("&transaction")
        account_menu=transaction_menu.addMenu("&account")
        debt_menu=transaction_menu.addMenu("&debt")
        statistics_menu=self.menuBar().addMenu("&statistics")

        #define menu actions
        branch_action=QAction("branch",self)
        branch_action.setStatusTip("branch basic information")
        branch_action.triggered.connect(self.branch_dialog)
        infrastracture_menu.addAction(branch_action)

        employee_action=QAction("employee",self)
        employee_action.setStatusTip("employee basic information ")
        employee_action.triggered.connect(self.employee_dialog)
        infrastracture_menu.addAction(employee_action)

        customer_action=QAction("customer",self)
        customer_action.setStatusTip("customer basic information")
        customer_action.triggered.connect(self.customer_dialog)
        client_menu.addAction(customer_action)

        responsible_action=QAction("responsible",self)
        responsible_action.setStatusTip("employee responsible for customer")
        responsible_action.triggered.connect(self.responsible_dialog)
        client_menu.addAction(responsible_action)

        saving_action=QAction("saving",self)
        saving_action.setStatusTip("saving account")
        saving_action.triggered.connect(self.saving_dialog)
        account_menu.addAction(saving_action)

        check_action=QAction("check",self)
        check_action.setStatusTip("check account")
        check_action.triggered.connect(self.check_dialog)
        account_menu.addAction(check_action)

        depositor_action=QAction("depositor",self)
        depositor_action.setStatusTip("customer owns account")
        depositor_action.triggered.connect(self.depositor_dialog)
        account_menu.addAction(depositor_action)

        loan_action=QAction("loan",self)
        loan_action.setStatusTip("branch give loan")
        loan_action.triggered.connect(self.loan_dialog)
        debt_menu.addAction(loan_action)

        payment_action=QAction("payment",self)
        payment_action.setStatusTip("payment for loan")
        payment_action.triggered.connect(self.payment_dialog)
        debt_menu.addAction(payment_action)

        borrow_action=QAction("borrow",self)
        borrow_action.setStatusTip("customer borrow loan")
        borrow_action.triggered.connect(self.borrow_dialog)
        debt_menu.addAction(borrow_action)

        statistics_overall_action=QAction("overall",self)
        statistics_overall_action.setStatusTip("overall statistics")
        statistics_overall_action.triggered.connect(self.statistics_overall_dialog)
        statistics_menu.addAction(statistics_overall_action)

        statistics_seasonly_action=QAction("seasonly",self)
        statistics_seasonly_action.setStatusTip("seasonly statistics")
        statistics_seasonly_action.triggered.connect(self.statistics_seasonly_dialog)
        statistics_menu.addAction(statistics_seasonly_action)

        statistics_monthly_action=QAction("monthly",self)
        statistics_monthly_action.setStatusTip("montly statistics")
        statistics_monthly_action.triggered.connect(self.statistics_monthly_dialog)
        statistics_menu.addAction(statistics_monthly_action)

        self.statusBar()
        #toolbar
        customer_toolbar=self.addToolBar("customer")
        customer_toolbar.addAction(customer_action)

        saving_toolbar=self.addToolBar("saving")
        saving_toolbar.addAction(saving_action)

        check_toolbar=self.addToolBar("check")
        check_toolbar.addAction(check_action)

        loan_toolbar=self.addToolBar("loan")
        loan_toolbar.addAction(loan_action)

        payment_toolbar=self.addToolBar("payment")
        payment_toolbar.addAction(payment_action)

        #central widgets
        left_layout=QVBoxLayout()
        title=QLabel("Bank Management System")
        title_font=title.font()
        title_font.setPointSize(16)
        title.setFont(title_font)
        author=QLabel("by drdh")
        version=QLabel("(v1.0)")

        #left_layout.addStretch(1)
        title_version_layout=QHBoxLayout()
        title_version_layout.addWidget(title)
        title_version_layout.addWidget(version)
        title_version_layout.addStretch()
        left_layout.addLayout(title_version_layout)
        left_layout.addWidget(author)
        left_layout.addStretch()

        
        register_title=QLabel("Administer Manage")
        register_title_font=register_title.font()
        register_title_font.setPointSize(13)
        register_title.setFont(register_title_font)
        self.userinput= QLineEdit()
        self.userinput.setPlaceholderText("Enter Username.")
        self.passinput = QLineEdit()
        self.passinput.setEchoMode(QLineEdit.Password)
        self.passinput.setPlaceholderText("Enter Password.")
        self.add_button=QPushButton("new")
        self.delete_button=QPushButton("delete")
        self.add_button.clicked.connect(self.add_user)
        self.delete_button.clicked.connect(self.delete_user)

        
        left_layout.addWidget(register_title)
        left_layout.addWidget(self.userinput)
        left_layout.addWidget(self.passinput)
        button_layout=QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        left_layout.addLayout(button_layout)

        #right_layout=QVBoxLayout()

        self.p = QPalette()
        self.pixmap = QPixmap("./img/bank4.jpg").scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.p.setBrush(QPalette.Background, QBrush(self.pixmap))
        self.setPalette(self.p)

        self.setStyleSheet("QToolBar {background:transparent}")


        layout=QHBoxLayout()
        layout.addLayout(left_layout)
        #layout.addLayout(right_layout)
        layout.addStretch(2)
        widget=QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(layout)

    def add_user(self):
        if self.userinput.text()!="" and self.passinput.text()!="":
            f=open("passwd.json","r")
            data=json.load(f)
            f.close()
            data[self.userinput.text()]=self.passinput.text()
            f=open("passwd.json","w")
            json.dump(data,f)
            f.close()
            QMessageBox.information(self, 'Succeed', "Add User or Change Password Successful")
        else:
            QMessageBox.warning(self, 'Error', "Empty Username or Password")
    
    def delete_user(self):
        f=open("passwd.json","r")
        data=json.load(f)
        f.close()
        if self.userinput.text() in data:
            if data[self.userinput.text()]!=self.passinput.text():
                QMessageBox.warning(self, 'Error', "Wrong Password")
            else:
                del data[self.userinput.text()]
                f=open("passwd.json","w")
                json.dump(data,f)
                f.close()
                QMessageBox.information(self, 'Succeed', "Delete User Successful")
        else:
            QMessageBox.warning(self, 'Error', "User doesn't Exist")
        
        
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

    def responsible_dialog(self):
        dlg=Responsible_Dialog()
        dlg.exec_()

    def saving_dialog(self):
        dlg=Saving_Dialog()
        dlg.exec_()
    
    def check_dialog(self):
        dlg=Check_Dialog()
        dlg.exec_()

    def depositor_dialog(self):
        dlg=Depositor_Dialog()
        dlg.exec_()

    def loan_dialog(self):
        dlg=Loan_Dialog()
        dlg.exec_()

    def payment_dialog(self):
        dlg=Payment_Dialog()
        dlg.exec_()

    def borrow_dialog(self):
        dlg=Borrow_Dialog()
        dlg.exec_()

    def statistics_overall_dialog(self):
        dlg=Statistics_Overall_Dialog()
        dlg.exec_()
    
    def statistics_seasonly_dialog(self):
        dlg=Statistics_Seasonly_Dialog()
        dlg.exec_()

    def statistics_monthly_dialog(self):
        dlg=Statistics_Monthly_Dialog()
        dlg.exec_()


app = QApplication(sys.argv)
passdlg = Login_Dialog()
if(passdlg.exec_() == 2): 
    window = Main_Window()
    window.show()
#    window.loaddata()
sys.exit(app.exec_())