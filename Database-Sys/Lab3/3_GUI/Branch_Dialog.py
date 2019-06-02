from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtSql import *
import sys,sqlite3,time

import os


class Branch_Dialog(QDialog):
    def __init__(self, *args, **kwargs):
        super(Branch_Dialog, self).__init__(*args, **kwargs)  
        self.setWindowTitle("Branch")
        #ui
        self.tableWidget = QTableView()

        if QSqlDatabase.contains('qt_sql_default_connection'):
            self.db=QSqlDatabase.database('qt_sql_default_connection')
        else:
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
        self.new_button=QPushButton("new")
        self.new_button.clicked.connect(self.new_event)
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
                QMessageBox.warning(QMessageBox(), 'Error', self.model.lastError().text())
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


