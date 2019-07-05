#ifndef LOGINDIALOG_H
#define LOGINDIALOG_H

#include <QDialog>
#include <QMessageBox>
#include <QtSql/QSqlDatabase>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QTcpSocket>

class AppPage;

class LoginDialog: public QDialog
{
    Q_OBJECT
public:
    LoginDialog(AppPage *page,QTcpSocket *socket,QWidget *parent = nullptr);
    ~LoginDialog();

private:
    QTcpSocket *sock;
    AppPage *page;

    QLabel *userNameLabel;
    QLabel *passwdLabel;
    QLineEdit *userNameEdit;
    QLineEdit *passwdEdit;
    QPushButton *connectButton;
    QPushButton *cancelButton;

private slots:
    void loginRequest();
    void loginReply();
    void closeDialog();
};

#endif // LOGINDIALOG_H
