#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <string>
#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

void test_trained_detector( String obj_det_filename, String test_dir, String videofilename );


void test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );

    vector< String > files;
    glob( test_dir, files );

    int delay = 0;
    VideoCapture cap;

    if ( videofilename != "" )
    {
        if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
            cap.open( videofilename[0] - '0' );
        else
            cap.open( videofilename );
    }

    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );

    for( size_t i=0;; i++ )
    {
        Mat img;

        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = imread( files[i] );
        }

        if ( img.empty() )
        {
            return;
        }

        vector< Rect > detections;
        vector< double > foundWeights;

        hog.detectMultiScale( img, detections, foundWeights );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
        }

        imshow( obj_det_filename, img );

        if( waitKey( delay ) == 27 )
        {
            return;
        }
    }
}

int main(int argc, char** argv){
    string MODEL_NAME = argv[1];
    string FILE_NAME = argv[2];

    // string MODEL_NAME = "./my_detector.yml";
    // string FILE_NAME = "./Video_1.avi";
    string FILE_DIR = "";
    
    test_trained_detector(MODEL_NAME, FILE_DIR, FILE_NAME);

    return 0;
}