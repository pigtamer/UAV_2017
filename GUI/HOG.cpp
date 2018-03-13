#pragma execution_character_set("utf-8")
#define _CRT_SECURE_NO_WARNINGS

#include "HOG.h"
#include "iostream"
#include "stdlib.h"
#include "string"
#include "time.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"

#include "QMessageBox"
#include "QtCore"
#include "qfiledialog.h"
#include "QtCore"

#include "train_utils.hpp"

using namespace cv::ml;
using namespace std;
using namespace cv;

HOG::HOG(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	connect(ui.play, SIGNAL(clicked()), this, SLOT(chooseModelPlay()));
	connect(ui.trainA, SIGNAL(clicked()), this, SLOT(trainModelA()));
	connect(ui.trainB, SIGNAL(clicked()), this, SLOT(trainModelB()));
}

void HOG::pixmapshow(cv::Mat frame)
{
	//这里将Mat转为QPixmap
	scene.clear();
	QImage picQImage;
	QPixmap picQPixmap;
	cv::cvtColor(frame, frame, CV_BGR2RGB);//三通道图片需bgr翻转成rgb
	picQImage = QImage((uchar*)frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
	picQPixmap = QPixmap::fromImage(picQImage);
	//scene.clear();
	scene.addPixmap(picQPixmap);
	ui.graphicsView->setScene(&scene);
	ui.graphicsView->show();
}

vector<float> HOG::get_svm_detector(const Ptr<SVM> &svm) 
{
// get the support vectors
Mat sv = svm->getSupportVectors();
const int sv_total = sv.rows;
// get the decision function
Mat alpha, svidx;
double rho = svm->getDecisionFunction(0, alpha, svidx);

CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
CV_Assert(sv.type() == CV_32F);

vector<float> hog_detector(sv.cols + 1);
memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
hog_detector[sv.cols] = (float)-rho;
return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning
* algorithms. TrainData is a matrix of size (#samples x max(#cols,#rows) per
* samples), in 32FC1. Transposition of samples are made if needed.
*/
void HOG::convert_to_ml(const vector<Mat> &train_samples, Mat &trainData) 
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < train_samples.size(); ++i) {
		CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);

		if (train_samples[i].cols == 1) {
			transpose(train_samples[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (train_samples[i].rows == 1) {
			train_samples[i].copyTo(trainData.row((int)i));
		}
	}
}

void HOG::load_images(const String &dirname, vector<Mat> &img_lst, bool showImages = false) 
{
	vector<String> files;
	glob(dirname, files);

	for (size_t i = 0; i < files.size(); ++i) {
		Mat img = imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}

		if (showImages) {
			imshow("image", img);
			waitKey(1);
		}
		img_lst.push_back(img);
	}
}

void HOG::sample_neg(const vector<Mat> &full_neg_lst, vector<Mat> &neg_lst, const Size &size) 
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < full_neg_lst.size(); i++)
		if (full_neg_lst[i].cols >= box.width &&
			full_neg_lst[i].rows >= box.height) 
		{
			box.x = rand() % (full_neg_lst[i].cols - size_x);
			box.y = rand() % (full_neg_lst[i].rows - size_y);
			Mat roi = full_neg_lst[i](box);
			neg_lst.push_back(roi.clone());
		}
}

void HOG::computeHOGs(const Size wsize, const vector<Mat> &img_lst,
	vector<Mat> &gradient_lst, bool use_flip) 
{
	HOGDescriptor hog;
	hog.winSize = wsize;
	Mat gray;
	vector<float> descriptors;

	for (size_t i = 0; i < img_lst.size(); i++) {
		if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height) {
			Rect r =
				Rect((img_lst[i].cols - wsize.width) / 2,
				(img_lst[i].rows - wsize.height) / 2, wsize.width, wsize.height);
			cvtColor(img_lst[i](r), gray, COLOR_BGR2GRAY);
			hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
			gradient_lst.push_back(Mat(descriptors).clone());
			if (use_flip) {
				flip(gray, gray, 1);
				hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
				gradient_lst.push_back(Mat(descriptors).clone());
			}
		}
	}
}

void HOG::test_trained_detector(String obj_det_filename,
	String test_video_dir,
	String videofilename) 
{
	//cout << "Testing trained detector..." << endl;
	ui.now->setText("Testing trained detector...");
	HOGDescriptor hog;
	hog.load(obj_det_filename);

	vector<String> files;
	glob(test_video_dir, files);

	int delay = 24;
	VideoCapture cap;

	if (videofilename != "") {
		if (videofilename.size() == 1 && isdigit(videofilename[0]))
			cap.open(videofilename[0] - '0');
		else
			cap.open(videofilename);
	}

	obj_det_filename = "testing " + obj_det_filename;
	namedWindow(obj_det_filename, WINDOW_NORMAL);

	for (size_t i = 0;; i++) {
		Mat img;

		if (cap.isOpened()) {
			cap >> img;
			delay = 1;
		}
		else if (i < files.size()) {
			img = imread(files[i]);
		}

		if (img.empty()) {
			return;
		}

		vector<Rect> detections;
		vector<double> foundWeights;

		hog.detectMultiScale(img, detections, foundWeights);
		if (foundWeights.size() > 0) {
			double MAX_WEIGHT = *max_element(std::begin(foundWeights), std::end(foundWeights));
			for (size_t j = 0; j < detections.size(); j++) {
				Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
				if (foundWeights[j] == MAX_WEIGHT) {
					if (foundWeights[j] > 0.1) {
						// cout << foundWeights[j] << endl;
						//显示SVM评分
						double w = foundWeights[j];
						putText(img, to_string(w), detections[j].tl(), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0));
						rectangle(img, detections[j], color, img.cols / 400 + 1);
					}
					else {
						//如果没有找到就显示！
						putText(img, "[!]", Point(0, img.cols / 4), FONT_HERSHEY_DUPLEX, 5, Scalar(0, 0, 255));
					}
				}
			}
		}
		else {
			putText(img, "[!]", Point(0, img.cols / 4), FONT_HERSHEY_DUPLEX, 5, Scalar(0, 0, 255));
		}
		//imshow(obj_det_filename, img);
		//改动1
		pixmapshow(img);
		if (waitKey(delay) == 27) {
			return;
		}

	}
}

int HOG::run(int detector_width, int detector_height, string pos_dir,
	string neg_dir, string test_video_dir,
	string obj_det_filename, string videofilename,
	bool FLAG_ONLY_TEST, bool FLAG_VISUALIZE_TRAIN,
	bool FLAG_FLIP_SAMPLES, bool FLAG_TRAIN_TWICE) 
{
	if (FLAG_ONLY_TEST) {
		test_trained_detector(obj_det_filename, test_video_dir, videofilename);
		exit(0);
	}

	if (pos_dir.empty() || neg_dir.empty()) {
		cout << "ILLEGAL DIR FOR SAMPLES." << endl;
		exit(1);
	}

	vector<Mat> pos_lst, full_neg_lst, neg_lst, gradient_lst;
	vector<int> labels;

	//clog << "Positive images are being loaded...";
	ui.now->setText("Positive images are being loaded...");
	load_images(pos_dir, pos_lst, FLAG_VISUALIZE_TRAIN);
	if (pos_lst.size() > 0) {
		//clog << "...[done]" << endl;
		ui.now->setText("...[done]");
	}
	else {
		//clog << "no image in " << pos_dir << endl;
		ui.now->setText("no image in posPATH");
		return 1;
	}

	Size pos_image_size = pos_lst[0].size();

	if (detector_width && detector_height) {
		pos_image_size = Size(detector_width, detector_height);
	}
	else {
		for (size_t i = 0; i < pos_lst.size(); ++i) {
			if (pos_lst[i].size() != pos_image_size) {
				cout << "All positive images should be same size!" << endl;
				exit(1);
			}
		}
		pos_image_size = pos_image_size / 8 * 8;
	}

	//clog << "Negative images are being loaded...";
	ui.now->setText("Negative images are being loaded...");
	load_images(neg_dir, full_neg_lst, false);
	sample_neg(full_neg_lst, neg_lst, pos_image_size);
	//clog << "...[done]" << endl;
	ui.now->setText("...[done]");
	//clog << "Histogram of Gradients are being calculated for positive images...";
	ui.now->setText("Histogram of Gradients are being calculated for positive images...");
	computeHOGs(pos_image_size, pos_lst, gradient_lst, FLAG_FLIP_SAMPLES);
	size_t positive_count = gradient_lst.size();
	labels.assign(positive_count, +1);
	//clog << "...[done] ( positive count : " << positive_count << " )" << endl;重定向
	ui.now->setText("...positive done");
	//clog << "Histogram of Gradients are being calculated for negative images...";
	ui.now->setText("Histogram of Gradients are being calculated for negative images...");
	computeHOGs(pos_image_size, neg_lst, gradient_lst, FLAG_FLIP_SAMPLES);
	size_t negative_count = gradient_lst.size() - positive_count;
	labels.insert(labels.end(), negative_count, -1);
	CV_Assert(positive_count < labels.size());
	//clog << "...[done] ( negative count : " << negative_count << " )" << endl;
	ui.now->setText("negative done");
	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	//clog << "Training SVM...";
	ui.now->setText("Training SVM");
	Ptr<SVM> svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(
		TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1);             // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01);            // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR;
								// // do regression task
	svm->train(train_data, ROW_SAMPLE, labels);
	//clog << "...[done]" << endl;
	ui.now->setText("...[done]");
	if (FLAG_TRAIN_TWICE) {
		//clog << "Testing trained detector on negative images. This may take a few minutes...";
			ui.now->setText("Testing trained detector on negative images. This may take a few minutes...");
			HOGDescriptor my_hog;
		my_hog.winSize = pos_image_size;

		// Set the trained svm to my_hog
		my_hog.setSVMDetector(get_svm_detector(svm));

		vector<Rect> detections;
		vector<double> foundWeights;

		for (size_t i = 0; i < full_neg_lst.size(); i++) {
			if (full_neg_lst[i].cols >= pos_image_size.width &&
				full_neg_lst[i].rows >= pos_image_size.height)
				my_hog.detectMultiScale(full_neg_lst[i], detections, foundWeights);
			else
				detections.clear();

			for (size_t j = 0; j < detections.size(); j++) {
				Mat detection = full_neg_lst[i](detections[j]).clone();
				cv::resize(detection, detection, pos_image_size, 0, 0, INTER_LINEAR_EXACT);
				neg_lst.push_back(detection);
			}

			if (FLAG_VISUALIZE_TRAIN) {
				for (size_t j = 0; j < detections.size(); j++) {
					rectangle(full_neg_lst[i], detections[j], Scalar(0, 255, 0), 2);
				}
				//imshow("testing trained detector on negative images", full_neg_lst[i]);
				//改动2
				pixmapshow(full_neg_lst[i]);
				waitKey(5);
			}
		}
		//clog << "...[done]" << endl;
		ui.now->setText("...[done]");
		gradient_lst.clear();
		//clog<< "Histogram of Gradients are being calculated for positive images...";
		ui.now->setText("Histogram of Gradients are being calculated for positive images...");
		computeHOGs(pos_image_size, pos_lst, gradient_lst, FLAG_FLIP_SAMPLES);
		positive_count = gradient_lst.size();
		//clog << "...[done] ( positive count : " << positive_count << " )" << endl;
		ui.now->setText("positive done");
		//clog<< "Histogram of Gradients are being calculated for negative images...";
		ui.now->setText("Histogram of Gradients are being calculated for negative images...");
		computeHOGs(pos_image_size, neg_lst, gradient_lst, FLAG_FLIP_SAMPLES);
		negative_count = gradient_lst.size() - positive_count;
		//clog << "...[done] ( negative count : " << negative_count << " )" << endl;
		ui.now->setText("negative done");
		labels.clear();
		labels.assign(positive_count, +1);
		labels.insert(labels.end(), negative_count, -1);

		//clog << "Training SVM again...";
		ui.now->setText("Training SVM again...");
		convert_to_ml(gradient_lst, train_data);
		svm->train(train_data, ROW_SAMPLE, labels);
		//clog << "...[done]" << endl;
		ui.now->setText("...[done]");
	}

	HOGDescriptor hog;
	hog.winSize = pos_image_size;
	hog.setSVMDetector(get_svm_detector(svm));
	hog.save(obj_det_filename);

	test_trained_detector(obj_det_filename, test_video_dir, videofilename);
	return 1;
}

// void HOG::pixmapshow(cv::Mat frame)


int HOG::chooseModelPlay()
{
	VIDEOPATH = QFileDialog::getOpenFileName(0, "Select Video", "", "", 0);
	MODELPATH = QFileDialog::getOpenFileName(0, "Select Model", "", "", 0);
	String obj_det_filename;
	String test_dir;
	String videofilename;
	
	obj_det_filename = MODELPATH.toLatin1().data();
	test_dir = "";
	videofilename = VIDEOPATH.toLatin1().data();

	HOGDescriptor hog;
	hog.load(obj_det_filename);

	vector< String > files;
	glob(test_dir, files);
	obj_det_filename = "testing " + obj_det_filename;

	cv::VideoCapture capture(VIDEOPATH.toLatin1().data());
	//打开视频文件
	if (!capture.isOpened())
	{
		return 1;
	}
	// 取得帧速率
	double rate = capture.get(CV_CAP_PROP_FPS);
	bool stop = false;
	cv::Mat frame; // 当前视频帧
	cv::namedWindow("demo", 0);
	cv::resizeWindow("demo", 1, 1);
	// 根据帧速率计算帧之间的等待时间，单位ms
	int delay = 1000 / rate;
	// 循环遍历视频中的全部帧
	while (!stop) {
		// 读取下一帧（如果有）

		if (!capture.read(frame))
			break;
		//
		Mat img;
		img = frame;
		vector< Rect > detections;
		vector< double > foundWeights;

		hog.detectMultiScale(img, detections, foundWeights);
		for (size_t j = 0; j < detections.size(); j++)
		{
			Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
			rectangle(img, detections[j], color, img.cols / 400 + 1);
		}
		//

		int WINDOW_SIZE[2]{ 480, 720 };
		double scale_ratio;
		cv::Size scaled_size;
		scale_ratio = std::min(double(WINDOW_SIZE[0]) / double(frame.rows), double(WINDOW_SIZE[1] / double(frame.cols)));
		scaled_size = cv::Size(scale_ratio * frame.cols, scale_ratio * frame.rows);
		cv::resize(frame, frame, scaled_size);
		pixmapshow(frame);
		// 等待一段时间，或者通过按键停止
		if (cv::waitKey(delay) >= 0)
			stop = true;
	}
	// 关闭视频文件
	// 不是必需的，因为类的析构函数会调用
	capture.release();
	return 1;
}

int HOG::trainModelA()
{
		VIDEOPATH = QFileDialog::getOpenFileName(0, "Select Video", "", "", 0);
	if (MODELPATH == NULL)
		MODELPATH = QFileDialog::getOpenFileName(0, "Select Model", "", "", 0);
	if (posPATH == NULL)
		posPATH = QFileDialog::getExistingDirectory(this, "Choose Pos Sample Dir", "./");
	if (negaPATH == NULL)
		negaPATH = QFileDialog::getExistingDirectory(this, "Choose Nega Sample Dir", "./");

	string modelpath = MODELPATH.toLatin1().data();
	string videopath = VIDEOPATH.toLatin1().data();
	string pospath = posPATH.toLatin1().data();
	string negapath = negaPATH.toLatin1().data();

	HOG::run(64, 64, pospath, negapath, "", modelpath, videopath, true, true);
	return 1;
}

int HOG::trainModelB()
{
		VIDEOPATH = QFileDialog::getOpenFileName(0, "Select Video", "", "", 0);
	if (MODELPATH == NULL)
		MODELPATH = QFileDialog::getOpenFileName(0, "Select Model", "", "", 0);
	if (posPATH == NULL)
		posPATH = QFileDialog::getExistingDirectory(this, "Choose Pos Sample Dir", "./");
	if (negaPATH == NULL)
		negaPATH = QFileDialog::getExistingDirectory(this, "Choose Nega Sample Dir", "./");

	string modelpath = MODELPATH.toLatin1().data();
	string videopath = VIDEOPATH.toLatin1().data();
	string pospath = posPATH.toLatin1().data();
	string negapath = negaPATH.toLatin1().data();

	/*HOG testTrainer;*/
	HOG::run(64, 64, pospath, negapath, "", modelpath, videopath,false, true);
	return 1;
}