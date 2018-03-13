#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <time.h>
#include <string>
using namespace cv;
using namespace cv::ml;
using namespace std;

class HoGTrainer{
private:
    string  SVM_TYPE;
    vector< float > get_svm_detector( const Ptr< SVM >& svm );
    void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
    void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
    void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
    void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
public:
     // original "main" func in train_HOG
    int run(int detector_width = 64, 
            int detector_height = 64, 
            string pos_dir = "./pos", 
            string neg_dir = "./nega", 
            string test_video_dir = "./testvid",
            string obj_det_filename = "./Model.yml",
            string videofilename = "",
            bool FLAG_ONLY_TEST = false, 
            bool FLAG_VISUALIZE_TRAIN = false, 
            bool FLAG_FLIP_SAMPLES = true, 
            bool FLAG_TRAIN_TWICE = true);

    void test_trained_detector( String obj_det_filename = "./Model.yml", String test_dir = "./testvid", String videofilename = "" );

};
