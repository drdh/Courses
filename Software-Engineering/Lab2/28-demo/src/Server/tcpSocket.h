#ifndef TCPSOCKET_H
#define TCPSOCKET_H

#include <QTcpSocket>
#include <QString>
//#include <tuple>
#include <QHostAddress>
#include <QByteArray>
#include <QtSql/QSqlDatabase>
#include <QSqlQuery>
#include <QFile>
#include <QProgressBar>
#include <QPushButton>

class Widget;

struct Request
{
    QHostAddress ip;
    int userId;
    int appId;
    QString appName;
    bool isDeveloper;
};

class TcpSocket : public QTcpSocket
{
    Q_OBJECT
public:
    TcpSocket(QSqlDatabase database,Widget *w);
    ~TcpSocket();
    Request req;

private:
    enum State{AnalyzeRequest,Upload};

    QSqlDatabase db;
    Widget *widget;
    void listApp();
    void listApp(QString appName);
    void getAppInfo(int appID);
    void login(QString userName,QString password);
    void signUp(QString userName,QString password);
    void download(int appID);
    void upload();
    void rcvFile();

    QString appPath = "/home/linan/Server/app/";            //此处后续应该改为相对路径
    QString iconPath = "/home/linan/Server/image/";
    int fileSize;
    int rcvSize;
    QFile *appFile = nullptr;

    State state;
    QByteArray rcvMsg;

    QVector<QString> appName;
    QVector<QProgressBar *> progressBar;
    QString developer;
    QVector<QPushButton *> passButton;

private slots:
    void clientDisconnectedSlot();
    void analyzeRequest();
    void addAppToDB();

signals:
    void newMsg(QString);
    void clientDisconnected(int);
};

#endif // TCPSOCKET_H
