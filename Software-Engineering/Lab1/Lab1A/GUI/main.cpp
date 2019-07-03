#include "widget.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QFile file(":/qss/style.qss");
    file.open(QFile::ReadOnly);
    Widget w;
    a.setStyleSheet(file.readAll());
    w.show();

    return a.exec();
}
