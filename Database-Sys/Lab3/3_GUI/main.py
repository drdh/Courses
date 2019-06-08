from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtSql import *
import sys,sqlite3,time

import os


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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('employee')
        self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
        self.model.setRelation(2, QSqlRelation("employee", "employee_id", "name"))
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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('responsible')
        self.model.setRelation(0, QSqlRelation("employee", "employee_id", "name"))
        self.model.setRelation(1, QSqlRelation("customer", "customer_id", "name"))
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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('saving_account')
        self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('check_account')
        self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
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

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db")

        self.model = QSqlRelationalTableModel()
        self.model.setTable('depositor')
        self.model.setRelation(0, QSqlRelation("account", "account_id", "account_id"))
        self.model.setRelation(1, QSqlRelation("customer", "customer_id", "name"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "account1")
        self.model.setHeaderData(1, Qt.Horizontal, "customer1")
        self.model.setHeaderData(2, Qt.Horizontal, "access_date")

        self.tableWidget.setModel(self.model)
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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('loan')
        self.model.setRelation(1, QSqlRelation("branch", "branch_name", "branch_name"))
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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('payment')
        self.model.setRelation(0, QSqlRelation("loan", "loan_id", "loan_id"))
        self.model.setEditStrategy(QSqlTableModel.OnFieldChange)

        self.model.setHeaderData(0, Qt.Horizontal, "loan_id")
        self.model.setHeaderData(1, Qt.Horizontal, "payment_id")
        self.model.setHeaderData(2, Qt.Horizontal, "payment_date")
        self.model.setHeaderData(3, Qt.Horizontal, "amount")

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

        self.model = QSqlRelationalTableModel()
        self.model.setTable('borrow')
        self.model.setRelation(0, QSqlRelation("customer", "customer_id", "name"))
        self.model.setRelation(1, QSqlRelation("loan", "loan_id", "loan_id"))
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

class Statistics_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Statistics_Dialog, self).__init__(*args, **kwargs) 
        self.setWindowTitle("Statistics") 
        #ui 
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
            self.db=QSqlDatabase.addDatabase('QSQLITE')
            self.db.setDatabaseName("../2_sqlite3_implement/data/test.db") 




class Main_Window(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Main_Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Bank Management System by drdh")
        self.setMinimumSize(800, 600)

        #define menu
        infrastracture_menu=self.menuBar().addMenu("&infrastructure")
        client_menu=self.menuBar().addMenu("&client")
        transaction_menu=self.menuBar().addMenu("&transaction")
        account_menu=transaction_menu.addMenu("&account")
        debt_menu=transaction_menu.addMenu("&debt")
        statistics_menu=self.menuBar().addMenu("&statistics")

        #define menu actions
        branch_action=QAction("branch",self)
        branch_action.triggered.connect(self.branch_dialog)
        infrastracture_menu.addAction(branch_action)

        employee_action=QAction("employee",self)
        employee_action.triggered.connect(self.employee_dialog)
        infrastracture_menu.addAction(employee_action)

        customer_action=QAction("customer",self)
        customer_action.triggered.connect(self.customer_dialog)
        client_menu.addAction(customer_action)

        responsible_action=QAction("responsible",self)
        responsible_action.triggered.connect(self.responsible_dialog)
        client_menu.addAction(responsible_action)

        saving_action=QAction("saving",self)
        saving_action.triggered.connect(self.saving_dialog)
        account_menu.addAction(saving_action)

        check_action=QAction("check",self)
        check_action.triggered.connect(self.check_dialog)
        account_menu.addAction(check_action)

        depositor_action=QAction("depositor",self)
        depositor_action.triggered.connect(self.depositor_dialog)
        account_menu.addAction(depositor_action)

        loan_action=QAction("loan",self)
        loan_action.triggered.connect(self.loan_dialog)
        debt_menu.addAction(loan_action)

        payment_action=QAction("payment",self)
        payment_action.triggered.connect(self.payment_dialog)
        debt_menu.addAction(payment_action)

        borrow_action=QAction("borrow",self)
        borrow_action.triggered.connect(self.borrow_dialog)
        debt_menu.addAction(borrow_action)

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

    def statistics_dialog(self):
        dlg=Statistics_Dialog()
        dlg.exec_()


app = QApplication(sys.argv)
passdlg = Login_Dialog()
if(passdlg.exec_() == 2): 
    window = Main_Window()
    window.show()
#    window.loaddata()
sys.exit(app.exec_())