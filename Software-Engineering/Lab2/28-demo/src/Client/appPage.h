#ifndef APPPAGE_H
#define APPPAGE_H

#include <QWidget>
#include <QToolButton>
#include <QLineEdit>
#include <QAction>
#include <QMenu>
#include <QVBoxLayout>
#include <QPushButton>
#include <QMainWindow>
#include <QTcpSocket>

class Client;

class AppPage : public QWidget
{
    Q_OBJECT

public:
    AppPage(Client *c,QMainWindow *parent = 0);
    ~AppPage();
    QAction *loginAction;
    Client *client;
    QTcpSocket *sock;

protected:
    QPushButton *optionsButton;
    QPushButton *userButton;
    QPushButton *backButton;
    QPushButton *searchButton;
    QMenu *options;
    QMenu *user;
    QLineEdit *searchBar;
    QAction *uploadAction;
    QAction *updateAction;
    QAction *signUpAction;

    QVBoxLayout *mainLayout;

private slots:
    void login();
    void signUp();
    void upload();
    virtual void analyzeReply() = 0;
};

#endif // APPPAGE
