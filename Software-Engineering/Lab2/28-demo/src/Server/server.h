#ifndef SERVER_H
#define SERVER_H

#include <QTcpServer>
#include <QList>
#include <widget.h>
#include <tcpSocket.h>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QMessageBox>

class Server : public QTcpServer
{
    Q_OBJECT

public:
    Server();
    ~Server();
    virtual void incomingConnection(qintptr descriptor);
    QList<TcpSocket *> sockets;
    QSqlDatabase db;

private:
    int port = 5678;
    Widget *w;

private slots:
    void deleteSocket(int descriptor);

signals:
    void newMsg(QString);
};

#endif // SERVER_H
