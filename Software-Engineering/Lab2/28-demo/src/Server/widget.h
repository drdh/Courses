#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QTextBrowser>
#include <QTableWidget>
#include <tuple>
#include <QProgressBar>
#include <QPushButton>

class Server;

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(Server *s,QWidget *parent = 0);
    ~Widget();
    void addApp(QString appName,QProgressBar *progressBar,QString developer,QPushButton *passButton);

private:
    QTextBrowser *connectionInfo;
    QTableWidget *appInfo;
    Server *server;

public slots:
    void update(QString req);
};

#endif // WIDGET_H
