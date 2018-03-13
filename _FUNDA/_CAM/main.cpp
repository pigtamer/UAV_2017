/*
REFERRED to a tracking demo on http://blog.csdn.net/dcrmg/article/details/52771372 , CSDN ID "牧野". SourceCode available on this site.
*/
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat img;
Mat rectImg;
Mat imgCopy;
bool leftButtonDownFlag = false;
cv::Point originalPoint;
cv::Point processPoint;

Mat targetImgHSV;
int histSize = 200;
float histR[] = {0,255}; //stands for histogram range
const float *histRange = histR;
int channels[] = {0,1};
Mat distHist;
Rect rect;
//vector<Point> pt; //Target trace 
void onMouse(int event,int x,int y,int flags ,void* arg0); //Mouse action traceback    

int main(int argc, char** argv)
{
	cv::VideoCapture cap(0);
    while(true){
       
        namedWindow("FRAMES",0);      
	    setMouseCallback("FRAMES",onMouse);
        
        if(!leftButtonDownFlag) //  
		{    
			cap >> img;//Import img;    
		}    
        
		if(!img.data)  
		{    
			break;    
		}   
		if(originalPoint!=processPoint&&!leftButtonDownFlag)  //Process selected rectangular area  
		{   
			Mat imgHSV;  
			Mat calcBackImage;  
			cvtColor(img,imgHSV,COLOR_RGB2HSV);  
			calcBackProject(&imgHSV,2,channels,distHist,calcBackImage,&histRange);  //Reversal Projector
			TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.001);    
			CamShift(calcBackImage, rect, criteria);     //CORE
			Mat imageROI=imgHSV(rect);   //Update current module        
			targetImgHSV=imgHSV(rect);  
			calcHist(&imageROI, 2, channels, Mat(), distHist, 1, &histSize, &histRange);    
			normalize(distHist, distHist, 0.0, 1.0, NORM_MINMAX);   //Normalize
			rectangle(img, rect, Scalar(255, 0, 0),3);    //Mark Target    
			// pt.push_back(Point(rect.x+rect.width/2,rect.y+rect.height/2));  
			// for(int i=0;i<pt.size()-1;i++)  
			// {  
			// 	cv::line(img,pt[i],pt[i+1],Scalar(0,255,0),2.5);  //Draw Target trace
			// }  
		}    

		imshow("FRAMES",img);   
		if(27 == waitKey(24)){
			break;
		}     
        
    }
    return 0;
}

//Mouse action traceback function 
void onMouse(int event,int x,int y,int flags,void *arg0)      
{     
	if(event==CV_EVENT_LBUTTONDOWN)      
	{      
		leftButtonDownFlag=true; //Left mouse clicked and hold on 
		originalPoint=Point(x,y);  //Start point of rectangle
		processPoint=originalPoint;    
	}      
	if(event==CV_EVENT_MOUSEMOVE&&leftButtonDownFlag)      
	{      
		imgCopy=img.clone();    
		processPoint=Point(x,y);    
		if(originalPoint!=processPoint)    
		{    
			//Mark selected range
			rectangle(imgCopy,originalPoint,processPoint,Scalar(255,0,0),2);    
		}    
		imshow("FRAMES",imgCopy);    
	}      
	if(event==CV_EVENT_LBUTTONUP)      
	{      
		leftButtonDownFlag=false;    
		rect=Rect(originalPoint,processPoint);        
		rectImg=img(rect); //shoe subplot
		imshow("Target Area",rectImg);        
		cvtColor(rectImg,targetImgHSV,COLOR_RGB2HSV);  
		calcHist(&targetImgHSV,2,channels,Mat(),distHist,1,&histSize,&histRange,true,false);
		normalize(distHist,distHist,0,255,CV_MINMAX);  
		imshow("Target HSV",targetImgHSV);  
	}        
}     