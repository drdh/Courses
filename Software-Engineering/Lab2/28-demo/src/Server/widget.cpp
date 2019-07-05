#include "widget.h"
#include <QVBoxLayout>
#include <QString>
#include <QStringList>
#include <QLabel>

Widget::Widget(Server *s, QWidget *parent)
    : QWidget(parent)
{
    setWindowState(Qt::WindowMaximized);
    connectionInfo = new QTextBrowser;
    appInfo = new QTableWidget;
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(connectionInfo);
    mainLayout->addWidget(appInfo);
    setLayout(mainLayout);

    server = s;

    appInfo->setColumnCount(4);
    QStringList header;
    header<<tr("Name")<<tr("Rate of Progress")<<tr("Developer")<<tr("Pass");
    appInfo->setHorizontalHeaderLabels(header);
}

Widget::~Widget()
{

}

void Widget::update(QString req)
{
    connectionInfo->append(req);
}

void Widget::addApp(QString appName, QProgressBar *progressBar, QString developer, QPushButton *passButton)
{
    int rowCount = appInfo->rowCount();
    appInfo->setRowCount(rowCount + 1);
    QLabel *nameLabel = new QLabel(appName,this);
    QLabel *developerLabel = new QLabel(developer,this);
    progressBar->show();
    appInfo->setCellWidget(rowCount,0,nameLabel);
    appInfo->setCellWidget(rowCount,1,progressBar);
    appInfo->setCellWidget(rowCount,2,developerLabel);
    appInfo->setCellWidget(rowCount,3,passButton);
}
