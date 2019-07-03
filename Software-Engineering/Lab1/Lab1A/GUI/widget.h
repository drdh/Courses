#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QTextEdit>
#include <QRadioButton>
#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QTextBrowser>
#include <QLayout>
#include <QLabel>
#include <QRegExp>
#include <QRegExpValidator>
#include <QFileDialog>
#include <QFile>
#include <QMessageBox>
#include <string>

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = 0);
    ~Widget();

private:
    QTextEdit *wordListField;
    QTextBrowser *resultField;
    QRadioButton *wOption;              //最多单词数量
    QRadioButton *cOption;              //最多字母个数
    QLineEdit *hLetterField;
    QLineEdit *tLetterField;
    QLineEdit *nField;
    QLabel *hLetterLabel;
    QLabel *tLetterLabel;
    QLabel *nLabel;

    QPushButton *checkButton;
    QPushButton *clearButton;
    QPushButton *findButton;
    QPushButton *saveButton;

private slots:
    void findFile();
    void saveFile();
    void checkInput();
    void clearIO();
};

#endif // WIDGET_H
