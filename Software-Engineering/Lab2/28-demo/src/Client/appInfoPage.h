#ifndef APPINFOPAGE_H
#define APPINFOPAGE_H

#include "appPage.h"
#include "appHomePage.h"
#include <QLabel>
#include <QTextBrowser>
#include <QProgressBar>
#include <QProgressDialog>

class AppInfoPage : public AppPage
{
    Q_OBJECT

public:
    AppInfoPage(Client *c,int appID,QString appName);
    ~AppInfoPage();

private:
    enum State{AnalyzeReply,GetAppInfo,DownLoad};

    State state;
    int appID;
    QString appName;

    QLabel *iconLabel;
    QLabel *nameLabel;
    QTextBrowser *introBrowser;
    QPushButton *downloadButton;          //中间的布局
    QProgressBar *progressBar;

    QListWidget *screenshots;
    QListWidget *comments;

    QByteArray rcvMsg;
    void getAppInfoRequest();
    void getAppInfoReply();
    void downloadReply();
    void rcvFile();

    QString appPath = "/home/linan/Client/app/";        //此处后续应该改为相对路径
    QString iconPath = "/home/linan/Client/image/";
    int fileSize = 0;
    int rcvSize = 0;
    QFile *appFile;

private slots:
    void analyzeReply();
    void backToHomePage();
    void downloadRequest();
};

#endif // APPINFOPAGE_H
