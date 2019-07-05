#include "client.h"
#include <QDebug>

Client::Client()
{
    setWindowState(Qt::WindowMaximized);

    serverIp.setAddress("127.0.0.1");
    socket = new QTcpSocket;
    connect(socket,SIGNAL(connected()),this,SLOT(hasConnected()));
    socket->connectToHost(serverIp,port);

    homePage = new AppHomePage(this);
    setCentralWidget(homePage);
    homePage->show();

    show();
}

Client::~Client()
{}

void Client::hasConnected()
{
    qDebug("has connected!\n");
}
