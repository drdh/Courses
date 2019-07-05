#include "appInfoPage.h"
#include "client.h"
#include <QPixmap>
#include <QImage>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QVBoxLayout>

AppInfoPage::AppInfoPage(Client *c, int appID, QString appName)
    :AppPage(c)
{
    if(client->userName != "")
    {
        loginAction->setDisabled(true);
        loginAction->setText(client->userName);
    }

    state = AnalyzeReply;
    this->appID = appID;
    this->appName = appName;

    iconLabel = new QLabel(this);
    iconLabel->setFixedSize(200,200);
    QPixmap pixmap;
    pixmap.convertFromImage(*(new QImage(iconPath + appID + ".png")));
    iconLabel->setPixmap(pixmap.scaled(200,200));

    nameLabel = new QLabel(appName,this);
    introBrowser = new QTextBrowser(this);
    introBrowser->setStyleSheet("font-size:20px;color:black");
    downloadButton = new QPushButton("Download");
    progressBar = new QProgressBar(this);
    appFile = new QFile;
    if(appFile->exists(appPath + appName))
    {
        downloadButton->hide();
        progressBar->setRange(0,1);
        progressBar->setValue(1);
        progressBar->show();
        delete appFile;
    }
    else progressBar->hide();

    QHBoxLayout *midLayout = new QHBoxLayout;
    midLayout->addWidget(iconLabel);
    QVBoxLayout *midLayout1 = new QVBoxLayout;
    nameLabel->setStyleSheet("font-size:40px;color:grey");
    midLayout1->addWidget(nameLabel);
    midLayout1->addWidget(introBrowser);
    QGridLayout *midLayout2 = new QGridLayout;
    midLayout2->addWidget(downloadButton,0,0,1,1);
    midLayout2->addWidget(progressBar,0,0,1,1);
    midLayout->addLayout(midLayout1);
    midLayout->addStretch();
    midLayout->addLayout(midLayout2);
    midLayout->setStretchFactor(iconLabel,2);
    midLayout->setStretchFactor(midLayout1,4);
    //midLayout->setStretchFactor(midLayout2,1);

    screenshots = new QListWidget(this);
    comments = new QListWidget(this);

    mainLayout->addLayout(midLayout);
    mainLayout->addWidget(screenshots);
    mainLayout->addWidget(comments);

    connect(backButton,SIGNAL(clicked(bool)),this,SLOT(backToHomePage()));
    connect(downloadButton,SIGNAL(clicked(bool)),this,SLOT(downloadRequest()));
    connect(sock,SIGNAL(readyRead()),this,SLOT(analyzeReply()));

    getAppInfoRequest();
}

AppInfoPage::~AppInfoPage()
{}

void AppInfoPage::backToHomePage()
{
    setParent(nullptr);
    client->setCentralWidget(client->homePage);
    if(client->userName != "")
    {
        client->homePage->loginAction->setDisabled(true);
        client->homePage->loginAction->setText(client->userName);
    }
    client->homePage->show();

    connect(sock,SIGNAL(readyRead()),client->homePage,SLOT(analyzeReply()));
    close();
}

void AppInfoPage::analyzeReply()
{
    rcvMsg = sock->readAll();
    if(state == AnalyzeReply)
    {
        if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "app info")
            getAppInfoReply();
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "app down")
        {
            fileSize = 0;
            rcvSize = 0;
            downloadReply();
        }
    }
    else if(state == GetAppInfo)
        getAppInfoReply();
    else if(state == DownLoad)
        rcvFile();
}

void AppInfoPage::getAppInfoRequest()
{
    sock->write((QString("app info") + QString::number(appID)).toUtf8());
}

void AppInfoPage::getAppInfoReply()
{
    int bytes = 8;
    int introSize = std::stoi(rcvMsg.mid(bytes,4).toStdString());
    QString introduction = QString::fromStdString(rcvMsg.mid(bytes + 4,introSize).toStdString());
    introBrowser->setText(introduction);
    state = AnalyzeReply;
}

void AppInfoPage::downloadRequest()
{
    sock->write((QString("app down") + QString::number(appID)).toUtf8());
}

void AppInfoPage::downloadReply()
{
    int bytes = 8;
    int appSize = std::stoi(rcvMsg.mid(bytes,8).toStdString(),0,16);
    fileSize = appSize;
    progressBar->setRange(0,fileSize);
    //qDebug("%d",appSize);
    QByteArray appData = rcvMsg.mid(bytes + 8);
    rcvSize = appData.size();
    qDebug("%s",appName.toStdString().data());
    appFile = new QFile;
    if(!appFile->exists(appPath + appName))
    {
        appFile->setFileName(appPath + appName);
        appFile->open(QIODevice::WriteOnly);
        appFile->write(appData);
        downloadButton->hide();
        progressBar->show();
        progressBar->setFormat("Downloading... %p%");
        progressBar->setValue(rcvSize);
        //使用QDataStream会在文件首部多4个字节，为文件大小
        if(appSize == rcvSize)
        {
            appFile->close();
            delete appFile;
            progressBar->setFormat("Downloaded");
            state = AnalyzeReply;
            return;
        }
        else state = DownLoad;
        return;
    }
    state = AnalyzeReply;
}

void AppInfoPage::rcvFile()
{
    QByteArray appData = rcvMsg;
    appFile->write(appData);
    rcvSize += appData.size();
    progressBar->setValue(rcvSize);
    if(rcvSize == fileSize)     //所有数据都已接收
    {
        appFile->close();
        delete appFile;
        progressBar->setFormat("Downloaded");
        state = AnalyzeReply;
    }
    else state = DownLoad;
}
