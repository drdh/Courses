#include "appHomePage.h"
#include "appInfoPage.h"
#include "client.h"
#include <QIcon>
#include <string>
#include <QDebug>
#include <QFont>

AppHomePage::AppHomePage(Client *c)
    :AppPage(c)
{   
    appArea = new QListWidget;
    appArea->setResizeMode(QListView::Adjust);
    appArea->setViewMode(QListView::IconMode);
    appArea->setMovement(QListView::Static);
    appArea->setIconSize(QSize(200,200));
    appArea->setSpacing(8);

    mainLayout->addWidget(appArea);

    state = AnalyzeReply;
    connect(appArea,SIGNAL(itemDoubleClicked(QListWidgetItem*)),this,SLOT(newAppInfoPage(QListWidgetItem*)));
    connect(sock,SIGNAL(readyRead()),this,SLOT(analyzeReply()));
    connect(searchButton,SIGNAL(clicked(bool)),this,SLOT(searchApp()));
    listAppRequest();
}

AppHomePage::~AppHomePage()
{}

void AppHomePage::analyzeReply()
{
    //qDebug("%d",sock->readAll().size());
    if(state == AnalyzeReply)
    {
        rcvMsg = sock->readAll();
        //qDebug("%d",rcvMsg.size());
        if(QString::fromStdString(rcvMsg.mid(0,4).toStdString()) == "list")
        {
            state = ListApp;
            listAppReply();
        }
    }
    else if(state == ListApp)
        listAppReply();
}

void AppHomePage::listAppRequest()
{
    sock->write("list all");
}

void AppHomePage::listAppReply()
{
    appArea->clear();
    appID.clear();
    appName.clear();
    icon.clear();
    item.clear();
    int bytes = 8;
    while(bytes < rcvMsg.size())
    {
        int id = std::stoi(rcvMsg.mid(bytes,8).toStdString()); //ID 4byte
        int nameSize = std::stoi(rcvMsg.mid(bytes + 8,2).toStdString(),0,16); //name size 2byte,用16进制表示，所以长度最大为FF，即255
        QString name = QString::fromStdString(rcvMsg.mid(bytes + 10,nameSize).toStdString());   //name
        int iconSize = std::stoi(rcvMsg.mid(bytes + 10 + nameSize,4).toStdString(),0,16);             //icon size 4byte
        QByteArray iconData = rcvMsg.mid(bytes + 14 + nameSize,iconSize);

        QFile newIcon;
        if(!newIcon.exists(iconPath + id + ".png"))
        {
            newIcon.setFileName(iconPath + id + ".png");
            newIcon.open(QIODevice::WriteOnly);
            newIcon.write(iconData);  //将图标暂存到本地
            //使用QDataStream会在文件首部多4个字节，为文件大小
            newIcon.close();
        }

        QListWidgetItem *newItem = new QListWidgetItem(QIcon(iconPath + id),name);
        newItem->setFont(QFont("Microsoft YaHei", 12, 50));

        appArea->addItem(newItem);
        appID.push_back(id);
        appName.push_back(name);
        icon.push_back(iconPath + id + ".png");
        item.push_back(newItem);

        bytes += 14 + nameSize + iconSize;
    }
    state = AnalyzeReply;
    //如果数据过多，分多次发送的话，这里需要记录已经接收的字节数是否等于数据总量，以决定状态的变化
}

void AppHomePage::searchApp()
{
    QString appName = searchBar->text();
    if(appName == "")listAppRequest();
    else sock->write(QString("list" + appName).toUtf8());
}

void AppHomePage::newAppInfoPage(QListWidgetItem *itemClicked)
{
    int i;
    for(i = 0; i < item.size(); i++)
        if(item[i] == itemClicked)break;
    //this->hide();
    disconnect(sock,SIGNAL(readyRead()),this,SLOT(analyzeReply()));
    //一定要断开连接，因为主界面并不会删除，当有新的数据到达时，会使得该对象中的槽也进行响应

    if(client->infoPage)
    {
        delete client->infoPage;
        client->infoPage = nullptr;     //删除某个对象不代表其指针值为0
    }
    client->infoPage = new AppInfoPage(client,appID[i],appName[i]);
    setParent(nullptr);
    client->setCentralWidget(client->infoPage);
}
