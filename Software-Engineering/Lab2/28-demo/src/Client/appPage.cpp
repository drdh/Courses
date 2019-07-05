#include "appPage.h"
#include "client.h"
#include "logindialog.h"
#include "signupdialog.h"
#include <QHBoxLayout>
#include <QFileDialog>
#include <QProgressDialog>

AppPage::AppPage(Client *c, QMainWindow *parent)
    :QWidget(parent)
{
    sock = c->socket;

    optionsButton = new QPushButton;
    optionsButton->setIcon(QIcon(":/icon/option.jpg"));
    optionsButton->setStyleSheet("QPushButton::menu-indicator{image:none;}");
    userButton = new QPushButton;
    userButton->setIcon(QIcon(":/icon/user.jpg"));
    userButton->setStyleSheet("QPushButton::menu-indicator{image:none;}");
    backButton = new QPushButton;
    backButton->setIcon(QIcon(":/icon/back.jpg"));
    searchButton = new QPushButton;
    searchButton->setText(tr("search"));

    options = new QMenu(tr("选项"));
    user = new QMenu(tr("用户"));

    searchBar = new QLineEdit;
    uploadAction = new QAction(tr("上传"));
    updateAction = new QAction(tr("更新"));
    loginAction = new QAction(tr("登录"));
    signUpAction = new QAction(tr("注册"));

    optionsButton->setMenu(options);
    userButton->setMenu(user);
    options->addAction(uploadAction);
    options->addAction(updateAction);
    user->addAction(loginAction);
    user->addAction(signUpAction);

    QHBoxLayout *upperLayout = new QHBoxLayout;
    mainLayout = new QVBoxLayout;
    upperLayout->addWidget(backButton);
    upperLayout->addWidget(searchBar);
    upperLayout->addWidget(searchButton);
    upperLayout->addWidget(userButton);
    upperLayout->addWidget(optionsButton);
    mainLayout->addLayout(upperLayout);
    setLayout(mainLayout);

    client = c;

    connect(loginAction,SIGNAL(triggered(bool)),this,SLOT(login()));
    connect(signUpAction,SIGNAL(triggered(bool)),this,SLOT(signUp()));
    connect(uploadAction,SIGNAL(triggered(bool)),this,SLOT(upload()));
}

AppPage::~AppPage()
{}

class Client;

void AppPage::login()
{
    new LoginDialog(this,client->socket,this);
}

void AppPage::signUp()
{
    new SignUpDialog(this,client->socket,this);
}

void AppPage::upload()
{
    if(!client->hasLogin)
        QMessageBox::information(this, tr("上传"),tr("请先登录"));
    else if(!client->isDeveloper)
        QMessageBox::information(this,tr("上传"),tr("您不是一位开发者"));
    else
    {
        QString fileName = QFileDialog::getOpenFileName(this,tr("选择应用"),"./");
        if(fileName.isEmpty())return;
        QString iconName = QFileDialog::getOpenFileName(this,tr("选择图片"),"./",tr("图片文件(*png)"));
        if(iconName.isEmpty())return;

        QFile iconFile(iconName);
        iconFile.open(QIODevice::ReadOnly);
        QFile file(fileName);
        file.open(QIODevice::ReadOnly);

        QByteArray dataToSend;
        QByteArray iconData = iconFile.readAll();       //图标采用和应用一样的名字
        QByteArray iconSizeData = QString::number(iconData.size(),16).toUtf8();
        iconSizeData.resize(4);

        QString name = fileName.section('/',-1,-1);
        QByteArray nameData = name.toUtf8();
        QByteArray nameSizeData = QString::number(nameData.size(),16).toUtf8();
        nameSizeData.resize(2);
        QByteArray userNameData = client->userName.toUtf8();
        QByteArray userNameSizeData = QString::number(userNameData.size(),16).toUtf8();
        userNameSizeData.resize(1);

        dataToSend.append("app send");
        dataToSend.append(nameSizeData);
        dataToSend.append(nameData);
        dataToSend.append(userNameSizeData);
        dataToSend.append(userNameData);
        dataToSend.append(iconSizeData);
        dataToSend.append(iconData);
        QByteArray fileData = file.readAll();
        QByteArray appSizeData = QString::number(fileData.size(),16).toUtf8();
        appSizeData.resize(8);
        dataToSend.append(appSizeData);

        QProgressDialog *progressDialog = new QProgressDialog("Uploading...","cancel",0,fileData.size(),this);
        progressDialog->hide();
        int blockSize = 64 * 1024;
        int sendTimes = fileData.size() / blockSize;
        for(int i = 0; i < sendTimes; i++)
        {
            if(!progressDialog->wasCanceled())
            {
                dataToSend.append(fileData.mid(blockSize * i,blockSize));
                sock->write(dataToSend);
                progressDialog->setValue(blockSize * i);
                dataToSend.clear();
            }
            else
            {
                dataToSend.clear();
                dataToSend.append("cancelup");
                sock->write(dataToSend);
            }
        }
        if(sendTimes * blockSize < fileData.size())
        {
            if(!progressDialog->wasCanceled())
            {
                dataToSend.append(fileData.mid(sendTimes * blockSize));
                sock->write(dataToSend);
                progressDialog->setValue(fileData.size());
            }
            else
            {
                dataToSend.append("cancelup");
                sock->write(dataToSend);
            }
        }
    }
}
