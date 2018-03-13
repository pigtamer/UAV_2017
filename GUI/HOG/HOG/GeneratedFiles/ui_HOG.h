/********************************************************************************
** Form generated from reading UI file 'HOG.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HOG_H
#define UI_HOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_HOGClass
{
public:
    QWidget *centralWidget;
    QGraphicsView *graphicsView;
    QPushButton *play;
    QPushButton *trainA;
    QPushButton *trainB;
    QLabel *label;
    QLabel *now;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *HOGClass)
    {
        if (HOGClass->objectName().isEmpty())
            HOGClass->setObjectName(QStringLiteral("HOGClass"));
        HOGClass->resize(770, 650);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(HOGClass->sizePolicy().hasHeightForWidth());
        HOGClass->setSizePolicy(sizePolicy);
        HOGClass->setMinimumSize(QSize(770, 650));
        HOGClass->setMaximumSize(QSize(770, 650));
        centralWidget = new QWidget(HOGClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        graphicsView = new QGraphicsView(centralWidget);
        graphicsView->setObjectName(QStringLiteral("graphicsView"));
        graphicsView->setGeometry(QRect(10, 0, 751, 491));
        play = new QPushButton(centralWidget);
        play->setObjectName(QStringLiteral("play"));
        play->setGeometry(QRect(510, 540, 51, 51));
        play->setCursor(QCursor(Qt::PointingHandCursor));
        QIcon icon;
        icon.addFile(QStringLiteral("ico/play.ico"), QSize(), QIcon::Normal, QIcon::Off);
        play->setIcon(icon);
        play->setIconSize(QSize(48, 48));
        trainA = new QPushButton(centralWidget);
        trainA->setObjectName(QStringLiteral("trainA"));
        trainA->setGeometry(QRect(600, 540, 51, 51));
        trainA->setCursor(QCursor(Qt::PointingHandCursor));
        QIcon icon1;
        icon1.addFile(QStringLiteral("ico/cube.ico"), QSize(), QIcon::Normal, QIcon::Off);
        trainA->setIcon(icon1);
        trainA->setIconSize(QSize(48, 48));
        trainB = new QPushButton(centralWidget);
        trainB->setObjectName(QStringLiteral("trainB"));
        trainB->setGeometry(QRect(680, 540, 51, 51));
        trainB->setCursor(QCursor(Qt::PointingHandCursor));
        QIcon icon2;
        icon2.addFile(QStringLiteral("ico/train.ico"), QSize(), QIcon::Normal, QIcon::Off);
        trainB->setIcon(icon2);
        trainB->setIconSize(QSize(48, 48));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 500, 61, 31));
        label->setStyleSheet(QLatin1String("background-color: rgb(11, 11, 11);\n"
"color: rgb(255, 255, 255);"));
        now = new QLabel(centralWidget);
        now->setObjectName(QStringLiteral("now"));
        now->setGeometry(QRect(70, 500, 691, 31));
        now->setFocusPolicy(Qt::NoFocus);
        now->setStyleSheet(QLatin1String("background-color: rgb(6, 6, 6);\n"
"color: rgb(244, 244, 244);"));
        HOGClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(HOGClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 770, 21));
        HOGClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(HOGClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        HOGClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(HOGClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        HOGClass->setStatusBar(statusBar);

        retranslateUi(HOGClass);

        QMetaObject::connectSlotsByName(HOGClass);
    } // setupUi

    void retranslateUi(QMainWindow *HOGClass)
    {
        HOGClass->setWindowTitle(QApplication::translate("HOGClass", "HOG", nullptr));
#ifndef QT_NO_TOOLTIP
        play->setToolTip(QApplication::translate("HOGClass", "\346\222\255\346\224\276", nullptr));
#endif // QT_NO_TOOLTIP
        play->setText(QString());
#ifndef QT_NO_SHORTCUT
        play->setShortcut(QApplication::translate("HOGClass", "Ctrl+Space", nullptr));
#endif // QT_NO_SHORTCUT
#ifndef QT_NO_TOOLTIP
        trainA->setToolTip(QApplication::translate("HOGClass", "\351\200\211\346\213\251\346\250\241\345\236\213", nullptr));
#endif // QT_NO_TOOLTIP
        trainA->setText(QString());
#ifndef QT_NO_SHORTCUT
        trainA->setShortcut(QApplication::translate("HOGClass", "Ctrl+M", nullptr));
#endif // QT_NO_SHORTCUT
#ifndef QT_NO_TOOLTIP
        trainB->setToolTip(QApplication::translate("HOGClass", "\350\256\255\347\273\203\346\250\241\345\236\213", nullptr));
#endif // QT_NO_TOOLTIP
        trainB->setText(QString());
#ifndef QT_NO_SHORTCUT
        trainB->setShortcut(QApplication::translate("HOGClass", "Ctrl+T", nullptr));
#endif // QT_NO_SHORTCUT
        label->setText(QApplication::translate("HOGClass", "\347\250\213\345\272\217\350\277\233\347\250\213:", nullptr));
        now->setText(QApplication::translate("HOGClass", "Waiting...", nullptr));
    } // retranslateUi

};

namespace Ui {
    class HOGClass: public Ui_HOGClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HOG_H
