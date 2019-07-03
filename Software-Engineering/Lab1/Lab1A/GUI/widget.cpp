#include "widget.h"
#include "../src/LongestWordChain.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
{
    findButton = new QPushButton(tr("&file"));
    saveButton = new QPushButton(tr("&save"));
    checkButton = new QPushButton(tr("check"));
    clearButton = new QPushButton(tr("&clear"));
    findButton->setToolTip(tr("Get the input from your file."));
    saveButton->setToolTip(tr("Save the result as your file."));
    checkButton->setToolTip(tr("Find the word list from the input."));
    clearButton->setToolTip(tr("Clear current input and result."));
    wordListField = new QTextEdit;
    wordListField->setPlaceholderText(tr("Enter your text..."));
    resultField = new QTextBrowser;
    QGridLayout *leftLayout = new QGridLayout;
    leftLayout->addWidget(wordListField,0,0,4,4);
    leftLayout->addWidget(resultField,4,0,3,4);
    leftLayout->addWidget(findButton,7,0,1,1);
    leftLayout->addWidget(saveButton,7,1,1,1);
    leftLayout->addWidget(clearButton,7,2,1,1);
    leftLayout->addWidget(checkButton,7,3,1,1);               //左边布局

    wOption = new QRadioButton(tr("most &words"));
    cOption = new QRadioButton(tr("most &letters"));
    wOption->setToolTip(tr("Find the word list with the most words."));
    cOption->setToolTip(tr("Find the word list with the most letters."));
    hLetterField = new QLineEdit;
    hLetterField->setToolTip(tr("Give a head letter."));
    hLetterLabel = new QLabel(tr("head letter"));
    tLetterField = new QLineEdit;
    tLetterField->setToolTip(tr("Give a tail letter."));
    tLetterLabel = new QLabel(tr("tail letter"));
    nField = new QLineEdit;
    nField->setToolTip(tr("Give the length of the word list."));
    nLabel = new QLabel(tr("number"));
    QGridLayout *rightLayout = new QGridLayout;
    rightLayout->addWidget(wOption,0,0,1,1);
    rightLayout->addWidget(cOption,1,0,1,1);
    rightLayout->addWidget(hLetterLabel,2,0,1,1);
    rightLayout->addWidget(hLetterField,2,1,1,1);
    rightLayout->addWidget(tLetterLabel,3,0,1,1);
    rightLayout->addWidget(tLetterField,3,1,1,1);
    rightLayout->addWidget(nLabel,4,0,1,1);
    rightLayout->addWidget(nField,4,1,1,1);             //右边布局

    QHBoxLayout *mainLayout = new QHBoxLayout;
    mainLayout->addLayout(leftLayout);
    mainLayout->addLayout(rightLayout);
    mainLayout->setStretchFactor(leftLayout,2);
    mainLayout->setStretchFactor(rightLayout,1);
    setLayout(mainLayout);                              //总体布局

    QRegExp letterRX("^[a-zA-Z]$");
    QRegExpValidator *letterValidator = new QRegExpValidator(letterRX,this);
    hLetterField->setValidator(letterValidator);
    tLetterField->setValidator(letterValidator);
    QRegExp numberRX("^[1-9][0-9]*$");
    QRegExpValidator *numberValidator = new QRegExpValidator(numberRX,this);
    nField->setValidator(numberValidator);              //限制输入格式

    connect(findButton,SIGNAL(clicked(bool)),this,SLOT(findFile()));
    connect(saveButton,SIGNAL(clicked(bool)),this,SLOT(saveFile()));
    connect(clearButton,SIGNAL(clicked(bool)),this,SLOT(clearIO()));
    connect(checkButton,SIGNAL(clicked(bool)),this,SLOT(checkInput()));         //连接信号和槽
}

Widget::~Widget()
{

}

void Widget::findFile()                 //从本地导入文件
{
    QString fileName = QFileDialog::getOpenFileName(this,tr("find source file"),"./",tr("Text(*.txt)"));
    if(fileName == "")return;
    int opt = QMessageBox::Yes;
    if(wordListField->toPlainText() != "")      //是否覆盖当前文本框中的文本
    {
        QMessageBox msgBox;
        msgBox.setText("Do you want to overwrite the current text?");
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        opt = msgBox.exec();
    }
    if(opt == QMessageBox::No)return;
    wordListField->clear();
    QFile file(fileName);
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QByteArray content = file.readAll();
        wordListField->setText(QString(content));
        resultField->clear();
        file.close();
    }

}

void Widget::saveFile()                 //保存检测结果
{
    if(resultField->toPlainText() == "")
    {
        QMessageBox msgBox;
        msgBox.setText("There is no result!");
        msgBox.exec();
        return;
    }
    QString filename = QFileDialog::getSaveFileName(this,tr("Save as"),"./",tr("Text(*.txt)"));
    if(filename == "")return;
    QFile file(filename);
    if(file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QString content = resultField->toPlainText();
        file.write(content.toUtf8());
        file.close();
    }
}

void Widget::checkInput()               //检测当前文本
{
    bool defineW = wOption->isChecked();
    bool defineC = cOption->isChecked();
    char specificHead = '\0',specificTail = '\0';
    int specificNum = 0;
    std::string text = wordListField->toPlainText().toStdString();
    //std::cout<<text<<endl;
    if(text == "")
    {
        QMessageBox msgBox;
        msgBox.setText("There is no input!");
        msgBox.exec();
        return;
    }
    if(!defineW && !defineC)
    {
        QMessageBox msgBox;
        msgBox.setText("Please check one button!");
        msgBox.exec();
        return;
    }
    QString hLetter = hLetterField->text();
    QString tLetter = tLetterField->text();
    QString num = nField->text();
    if(hLetter != "")specificHead = hLetter.at(0).toLatin1();
    if(tLetter != "")specificTail = tLetter.at(0).toLatin1();
    if(num != "")specificNum = num.toInt();             //分析各个窗口部件，得到参数
    if(specificNum == 1)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("The number of the words must be larger than 1!"));
        msgBox.exec();
        return;
    }
    //std::cout<<text<<std::endl;
    try
    {
        std::string result = LWC(defineW,text,specificNum,specificHead,specificTail);
        //std::cout<<result<<std::endl;
        resultField->clear();
        resultField->setText(QString::fromStdString(result));
    }
    catch(const char *msg)
    {
        QMessageBox msgBox;
        msgBox.setText(QString(QLatin1String(msg)));
        msgBox.exec();
        return;
    }

}

void Widget::clearIO()              //清空文本框的输入和输出
{
    wordListField->clear();
    resultField->clear();
}
