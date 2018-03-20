# UAV 2017

[这是2017年“移动相机下运动目标检测”的项目仓库](https://github.com/pigtamer/UAV_2017)

* _FUNDA：基础方法，包含光流、差分法、均值漂移法的实验结果
* HOG2D：运用二维方向梯度直方图的检测
  * _INTF_TRAIN: 训练分类器，train.exe是opencv340.dll编译的可执行文件
  * exSample.py: 从视频序列抽出样本的python脚本
* tryHOG3D：运用三维方向梯度直方图特征的目标检测测试，基于Python尚未完成
* GUI：图形界面，由Qt5开发，在MSVC15 2017 编译通过