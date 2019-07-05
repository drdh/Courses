#include "signupdialog.h"
#include "appPage.h"
#include "client.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>


SignUpDialog::SignUpDialog(AppPage *page, QTcpSocket *socket, QWidget *parent)
    :QDialog(parent)
{
    setAttribute(Qt::WA_DeleteOnClose);
    sock = socket;
    this->page = page;

    userNameLabel = new QLabel(tr("用户名"));
    passwdLabel = new QLabel(tr("密码"));
    ensurePasswdLabel = new QLabel(tr("确认密码"));
    userNameEdit = new QLineEdit;
    passwdEdit = new QLineEdit;
    ensurePasswdEdit = new QLineEdit;
    passwdEdit->setEchoMode(QLineEdit::Password);
    ensurePasswdEdit->setEchoMode(QLineEdit::Password);
    connectButton = new QPushButton(tr("注册"));
    cancelButton = new QPushButton(tr("取消"));

    QGridLayout *upLayout = new QGridLayout;
    upLayout->addWidget(userNameLabel,0,0,1,1);
    upLayout->addWidget(userNameEdit,0,1,1,1);
    upLayout->addWidget(passwdLabel,1,0,1,1);
    upLayout->addWidget(passwdEdit,1,1,1,1);
    upLayout->addWidget(ensurePasswdLabel,2,0,1,1);
    upLayout->addWidget(ensurePasswdEdit,2,1,1,1);

    QHBoxLayout *downLayout = new QHBoxLayout;
    downLayout->addWidget(connectButton);
    downLayout->addWidget(cancelButton);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addLayout(upLayout);
    mainLayout->addLayout(downLayout);

    setLayout(mainLayout);
    show();

    connect(connectButton,SIGNAL(clicked(bool)),this,SLOT(signUpRequest()));
    connect(cancelButton,SIGNAL(clicked(bool)),this,SLOT(closeDialog()));
    disconnect(sock,SIGNAL(readyRead()),0,0);
    connect(sock,SIGNAL(readyRead()),this,SLOT(signUpReply()));
}

SignUpDialog::~SignUpDialog()
{
    connect(page->client->socket,SIGNAL(readyRead()),page,SLOT(analyzeReply()));
    disconnect(sock,SIGNAL(readyRead()),this,SLOT(signUpReply()));
}

void SignUpDialog::signUpRequest()
{
    if(passwdEdit->text() != ensurePasswdEdit->text())
    {
        QMessageBox::information(this, tr("注册失败"),tr("The passwords are inconsistent!"));
        return;
    }
    QByteArray dataToSend;
    QByteArray reqData = QString("signup  ").toUtf8();
    QByteArray userName = userNameEdit->text().toUtf8();
    QByteArray password = passwdEdit->text().toUtf8();
    QByteArray userNameSize = QString::number(userName.size(),16).toUtf8();
    userNameSize.resize(1);
    QByteArray passwordSize = QString::number(password.size(),16).toUtf8();
    passwordSize.resize(1);

    dataToSend.append(reqData);
    dataToSend.append(userNameSize);
    dataToSend.append(userName);
    dataToSend.append(passwordSize);
    dataToSend.append(password);

    sock->write(dataToSend);
}

void SignUpDialog::signUpReply()
{
    QByteArray rcvMsg = sock->readAll();
    int bytes = 8;

    if(rcvMsg.mid(bytes,1).toStdString().data()[0] == '1')
        close();
    else
    {
        int errorSize = std::stoi(rcvMsg.mid(bytes + 1,2).toStdString(),0,16);
        QString error = QString::fromStdString(rcvMsg.mid(bytes + 3,errorSize).toStdString());
        QMessageBox::information(this, tr("注册失败"),error);
    }
}

void SignUpDialog::closeDialog()
{
    disconnect(page->sock,SIGNAL(readyRead()),this,SLOT(signUpReply()));
    close();
}
