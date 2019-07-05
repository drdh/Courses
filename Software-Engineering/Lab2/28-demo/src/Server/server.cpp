#include "server.h"
#include <QHostAddress>
#include <QDateTime>
#include <QDebug>

Server::Server()
{
    db = QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName("127.0.0.1");
    db.setDatabaseName("app store");
    db.setPort(3306);
    db.setUserName("root");
    db.setPassword("linan19980922");
    w = new Widget(this);
    if(!db.open())
    {
        QMessageBox::information(w, tr("app store"),db.lastError().text());
        //exit(0);
    }
    w->show();
    listen(QHostAddress::Any,port);
}

Server::~Server()
{}

void Server::incomingConnection(qintptr descriptor)     //如果是int则不会调用该函数
{
    connect(this,SIGNAL(newMsg(QString)),w,SLOT(update(QString)));

    TcpSocket *newSocket = new TcpSocket(db,w);
    newSocket->setSocketDescriptor(descriptor);
    sockets.append(newSocket);

    emit(newMsg(QDateTime::currentDateTime().toString() + " " + newSocket->peerAddress().toString() + " has connected."));

    connect(newSocket,SIGNAL(clientDisconnected(int)),this,SLOT(deleteSocket(int)));
    connect(newSocket,SIGNAL(newMsg(QString)),w,SLOT(update(QString)));
}

void Server::deleteSocket(int descriptor)
{
    for(QList<TcpSocket *>::iterator it = sockets.begin(); it != sockets.end(); it++)
    {
        if((*it)->socketDescriptor() == descriptor)
        {
            sockets.removeOne(*it);
            emit(newMsg(QDateTime::currentDateTime().toString() + "" + (*it)->peerAddress().toString() + "has disconnected."));
            //delete (*it);
            return;
        }
    }
}
