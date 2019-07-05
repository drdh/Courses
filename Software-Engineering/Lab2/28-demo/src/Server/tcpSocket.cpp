#include "tcpSocket.h"
#include <QDateTime>
#include <string>
#include <QDebug>
#include <widget.h>
#include <QVariant>

TcpSocket::TcpSocket(QSqlDatabase database,Widget *w)
{
    db = database;
    widget = w;
    state = AnalyzeRequest;
    req.ip = peerAddress();
    connect(this,SIGNAL(readyRead()),this,SLOT(analyzeRequest()));
    connect(this,SIGNAL(disconnected()),this,SLOT(clientDisconnectedSlot()));
}

TcpSocket::~TcpSocket()
{}

void TcpSocket::clientDisconnectedSlot()
{
    emit clientDisconnected(socketDescriptor());
}

void TcpSocket::analyzeRequest()
{
    /*QString msg = QDateTime::currentDateTime().toString() + '\t' + req.userId + '\t' + req.appId
            + '\t' + req.appName + '\t' + req.isDeveloper + '\n';
    emit(newMsg(msg));*/
    rcvMsg = readAll();
    if(state == AnalyzeRequest)
    {
        if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "list all")
            listApp();
        else if(QString::fromStdString(rcvMsg.mid(0,4).toStdString()) == "list")
        {
            QString appName = QString::fromStdString(rcvMsg.mid(4).toStdString());
            listApp(appName);
        }
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "app info")
        {
            int appID = std::stoi(rcvMsg.mid(8).toStdString());
            getAppInfo(appID);
        }
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "login   ")
        {
            int userNameSize = std::stoi(rcvMsg.mid(8,1).toStdString(),0,16);
            QString userName = QString::fromStdString(rcvMsg.mid(9,userNameSize).toStdString());
            int passwordSize = std::stoi(rcvMsg.mid(9 + userNameSize,1).toStdString(),0,16);
            QString password = QString::fromStdString(rcvMsg.mid(10 + userNameSize,passwordSize).toStdString());
            login(userName,password);
        }
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "signup  ")
        {
            int userNameSize = std::stoi(rcvMsg.mid(8,1).toStdString(),0,16);
            QString userName = QString::fromStdString(rcvMsg.mid(9,userNameSize).toStdString());
            int passwordSize = std::stoi(rcvMsg.mid(9 + userNameSize,1).toStdString(),0,16);
            QString password = QString::fromStdString(rcvMsg.mid(10 + userNameSize,passwordSize).toStdString());
            signUp(userName,password);
        }
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "app down")
        {
            int appID = std::stoi(rcvMsg.mid(8).toStdString());
            download(appID);
        }
        else if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "app send")
        {
            fileSize = 0;
            rcvSize = 0;
            upload();
        }
    }
    else if(state == Upload)
        rcvFile();
}

void TcpSocket::listApp()
{
    QByteArray dataToSend;
    dataToSend.append("list all");
    QSqlQuery query;
    query.exec("select `app store`.`Application`.`App ID`,`app store`.`Application`.`App Name`,"
               "`app store`.`Application`.`Icon` from `app store`.`Application`");
    while(query.next())
    {
        QByteArray idData = QString::number(query.value(0).toInt()).toUtf8();
        idData.resize(8);
        QByteArray nameData = (query.value(1).toString()).toUtf8();
        QByteArray nameSizeData = QString::number(nameData.size(),16).toUtf8();
        nameSizeData.resize(2);
        QByteArray iconData = query.value(2).toByteArray();
        QByteArray iconSizeData = QString::number(iconData.size(),16).toUtf8();
        iconSizeData.resize(4);

        dataToSend.append(idData);      //8B
        dataToSend.append(nameSizeData);        //1B
        dataToSend.append(nameData);
        dataToSend.append(iconSizeData);    //4B
        dataToSend.append(iconData);
    }
    write(dataToSend);
}

void TcpSocket::listApp(QString appName)
{
    QByteArray dataToSend;
    dataToSend.append("list one");
    QSqlQuery query;
    query.exec("select `app store`.`Application`.`App ID`,`app store`.`Application`.`App Name`,"
               "`app store`.`Application`.`Icon` from `app store`.`Application` where `Application`.`App Name` "
               "like '%" + appName + "%'");

    while(query.next())
    {
        QByteArray idData = QString::number(query.value(0).toInt()).toUtf8();
        idData.resize(8);

        QByteArray nameData = (query.value(1).toString()).toUtf8();

        QByteArray nameSizeData = QString::number(nameData.size(),16).toUtf8();
        nameSizeData.resize(2);

        QByteArray iconData = query.value(2).toByteArray();

        QByteArray iconSizeData = QString::number(iconData.size(),16).toUtf8();
        iconSizeData.resize(4);

        dataToSend.append(idData);      //8B
        dataToSend.append(nameSizeData);        //1B
        dataToSend.append(nameData);
        dataToSend.append(iconSizeData);    //4B
        dataToSend.append(iconData);
    }
    write(dataToSend);
}

void TcpSocket::getAppInfo(int appID)
{
    QByteArray dataToSend;
    dataToSend.append("app info");
    QSqlQuery query;
    query.exec("select `app store`.`Application`.`Introduction` from `app store`.`Application` where "
               "`Application`.`App ID` = " + QString::number(appID));
    while(query.next())
    {
        QByteArray introData = (query.value(0).toString()).toUtf8();

        QByteArray introSizeData = QString::number(introData.size()).toUtf8();
        introSizeData.resize(4);
        //qDebug("%d",introData.size());
        //qDebug("%s",introData.toStdString().data());
        dataToSend.append(introSizeData);      //4B
        dataToSend.append(introData);
    }
    write(dataToSend);
}

void TcpSocket::login(QString userName, QString password)
{
    QByteArray dataToSend;
    QByteArray success;
    QByteArray isDeveloper;
    QByteArray errorMsg;
    QByteArray errorMsgSize;
    dataToSend.append("login   ");
    QSqlQuery query;
    query.exec("select `Client`.`Password`,`Client`.`Is Developer` from `app store`.`Client` where "
               "`Client`.`ID` = '" + userName + "'");
    if(!query.next())
    {
        success = QByteArray(1,'0');
        QString error = "The user does not exist!";
        errorMsg = error.toUtf8();
        errorMsgSize = QString::number(errorMsg.size(),16).toUtf8();
        dataToSend.append(success);
        dataToSend.append(errorMsgSize);
        dataToSend.append(errorMsg);
    }
    else if(query.value(0).toString() != password)
    {
        success = QByteArray(1,'0');
        QString error = "The password is wrong!";
        errorMsg = error.toUtf8();
        errorMsgSize = QString::number(errorMsg.size(),16).toUtf8();
        dataToSend.append(success);
        dataToSend.append(errorMsgSize);
        dataToSend.append(errorMsg);
    }
    else
    {
        success = QByteArray(1,'1');
        if(query.value(1).toInt() == 1)isDeveloper = QByteArray(1,'1');
        else isDeveloper = QByteArray(1,'0');
        dataToSend.append(success);
        dataToSend.append(isDeveloper);
        /*query.exec("update `App Store`.`Client` set `Client`.`Is Online` = 1 where "
                   "`Client`.`ID` = '" + userName + "'");*/
    }
    write(dataToSend);
}

void TcpSocket::signUp(QString userName, QString password)
{
    QByteArray dataToSend;
    QByteArray success;
    QByteArray errorMsg;
    QByteArray errorMsgSize;
    dataToSend.append("login   ");
    QSqlQuery query;
    query.exec("select * from `app store`.`Client` where "
               "`Client`.`ID` = '" + userName + "'");
    if(query.next())
    {
        success = QByteArray(1,'0');
        QString error = "The username has be used!";
        errorMsg = error.toUtf8();
        errorMsgSize = QString::number(errorMsg.size(),16).toUtf8();
        dataToSend.append(success);
        dataToSend.append(errorMsgSize);
        dataToSend.append(errorMsg);
    }
    else
    {
        success = QByteArray(1,'1');
        dataToSend.append(success);
        query.exec("insert into `app store`.`Client`(`Client`.`ID`,`Client`.`Password`,`Client`.`Is Developer`)"
                   " values('" + userName + "','" + password + "',0)");
    }
    write(dataToSend);
}

void TcpSocket::download(int appID)
{
    QByteArray dataToSend;
    dataToSend.append("app down");
    QSqlQuery query;
    query.exec("select `Application`.`Package` from `app store`.`Application` where "
               "`Application`.`App ID` = " + QString::number(appID));

    query.next();

    QByteArray appData = query.value(0).toByteArray();
    QByteArray appSizeData = QString::number(appData.size(),16).toUtf8();
    appSizeData.resize(8);

    dataToSend.append(appSizeData);

    int blockSize = 64 * 1024;
    int sendTimes = appData.size() / blockSize;
    for(int i = 0; i < sendTimes; i++)
    {
        dataToSend.append(appData.mid(blockSize * i,blockSize));
        write(dataToSend);
        dataToSend.clear();
    }
    if(sendTimes * blockSize < appData.size())
    {
        dataToSend.append(appData.mid(sendTimes * blockSize));
        write(dataToSend);
    }
}

void TcpSocket::upload()
{
    int bytes = 8;
    int nameSize = std::stoi(rcvMsg.mid(bytes,2).toStdString(),0,16);
    QString name = QString::fromStdString(rcvMsg.mid(bytes + 2,nameSize).toStdString());

    int userNameSize = std::stoi(rcvMsg.mid(bytes + 2 + nameSize,1).toStdString(),0,16);
    QString userName = QString::fromStdString(rcvMsg.mid(bytes + 3 + nameSize,userNameSize).toStdString());

    int iconSize = std::stoi(rcvMsg.mid(bytes + 3 + nameSize + userNameSize,4).toStdString(),0,16);

    QFile iconFile;
    if(!iconFile.exists(iconPath + name + "_" + userName))
    {
        iconFile.setFileName(iconPath + name + "_" + userName);
        iconFile.open(QIODevice::WriteOnly);
        iconFile.write(rcvMsg.mid(bytes + 7 + nameSize + userNameSize,iconSize));
        iconFile.close();
    }

    int appSize = std::stoi(rcvMsg.mid(bytes + 7 + nameSize + userNameSize + iconSize,8).toStdString(),0,16);
    fileSize = appSize;
    QByteArray appData = rcvMsg.mid(bytes + 15 + nameSize + userNameSize + iconSize);
    rcvSize = appData.size();

    //这里尚未处理当开发者取消上传时控件从vector中的删除
    appName.push_back(name);
    QProgressBar *newProgressBar = new QProgressBar;
    progressBar.push_back(newProgressBar);
    QPushButton *newPassButton = new QPushButton("pass");
    passButton.push_back(newPassButton);
    newPassButton->setChecked(false);
    connect(newPassButton,SIGNAL(clicked(bool)),this,SLOT(addAppToDB()));
    newProgressBar->setRange(0,fileSize);
    newProgressBar->setFormat("Receiving... %p%");
    developer = userName;
    widget->addApp(name,newProgressBar,developer,newPassButton);


    appFile = new QFile;
    if(!appFile->exists(appPath + name + "_" + userName))
    {
        appFile->setFileName(appPath + name + "_" + userName);
        appFile->open(QIODevice::WriteOnly);
        appFile->write(appData);
        newProgressBar->setValue(rcvSize);
        //使用QDataStream会在文件首部多4个字节，为文件大小
        if(appSize == rcvSize)
        {
            appFile->close();
            delete appFile;
            newProgressBar->setFormat("Received %p%");
            state = AnalyzeRequest;
            return;
        }
        else state = Upload;
        return;
    }
    newProgressBar->setValue(fileSize);
    newProgressBar->setFormat("Received %p%");
    state = AnalyzeRequest;
}

void TcpSocket::rcvFile()
{
    if(QString::fromStdString(rcvMsg.mid(0,8).toStdString()) == "cancelup")
    {
        fileSize = 0;
        rcvSize = 0;
        delete appFile;
        state = AnalyzeRequest;
        return;
    }
    QByteArray appData = rcvMsg;
    appFile->write(appData);
    rcvSize += appData.size();
    progressBar.last()->setValue(rcvSize);
    if(rcvSize == fileSize)     //所有数据都已接收
    {
        appFile->close();
        delete appFile;
        progressBar.last()->setFormat("Received %p%");
        state = AnalyzeRequest;
    }
    else state = Upload;
}

void TcpSocket::addAppToDB()
{
    QPushButton *sender = qobject_cast<QPushButton *>(QObject::sender());
    int i;
    for(i = 0; i < passButton.size(); i++)
        if(sender == passButton[i])break;

    QSqlQuery query;
    query.exec("select max(`Application`.`App ID`) from `app store`.`Application`");
    query.next();
    int appID = query.value(0).toInt();

    QString app = appPath + appName[i] + "_" + developer;
    QString icon = iconPath + appName[i] + "_" + developer;
    QFile file(app);
    QFile iconFile(icon);
    file.open(QIODevice::ReadOnly);
    iconFile.open(QIODevice::ReadOnly);
    QVariant iconData(iconFile.readAll());
    QVariant appData(file.readAll());
    file.close();
    iconFile.close();

    query.clear();
    query.prepare("insert into `app store`.`Application` (`App ID`,`App Name`,`Icon`,`Package`,`Developer ID`)"
                  " values(?,?,?,?,?)");
    query.bindValue(0,appID + 1);
    query.bindValue(1,appName[i]);
    query.bindValue(2,iconData);
    query.bindValue(3,appData);
    query.bindValue(4,developer);
    query.exec();
    sender->setDisabled(true);
    sender->setText("passed");
}
