#ifndef SIGNUP_H
#define SIGNUP_H

#include <QDialog>
#include <QMessageBox>
#include <QtSql/QSqlDatabase>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <QTcpSocket>

class AppPage;

class SignUpDialog: public QDialog
{
    Q_OBJECT
public:
    SignUpDialog(AppPage *page,QTcpSocket *socket,QWidget *parent = nullptr);
    ~SignUpDialog();

private:
    QTcpSocket *sock;
    AppPage *page;

    QLabel *userNameLabel;
    QLabel *passwdLabel;
    QLabel *ensurePasswdLabel;
    QLineEdit *userNameEdit;
    QLineEdit *passwdEdit;
    QLineEdit *ensurePasswdEdit;
    QPushButton *connectButton;
    QPushButton *cancelButton;

private slots:
    void signUpRequest();
    void signUpReply();
    void closeDialog();
};

#endif // SIGNUP_H
