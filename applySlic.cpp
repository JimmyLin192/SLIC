// c++ standard headers
#include <cstdlib>
#include <cstdio> 
#include <iostream> 
#include <fstream>
#include <iomanip>
#include <map> 
// eigen matrix library headers
#include "Eigen/Core"

// opencv library header
#include "cv.h" 
#include "cxcore.h" 
#include "highgui.h"

// darwin library header
#include "drwnBase.h" 
#include "drwnIO.h" 
#include "drwnML.h" 
#include "drwnVision.h"

#include "malloc.h"
#include "slic.h"

using namespace std;
using namespace Eigen;

// HELP FUNCTION
void usage() {
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./slic [OPTIONS] <inputDir> <outputDir>\n";
    cerr << "OPTIONS:\n"
        << "  -k             :: number of clusters specified (15 by default)\n"
        << "  -t             :: threshold (1e-3 by default)\n"
        << "  -m             :: relative importance of two types of distances\n"
        << "  -x             :: visualize\n"
        << DRWN_STANDARD_OPTIONS_USAGE
        << endl;
}

// MAIN FUNCTION
int main (int argc, char * argv []) {
   bool bVisualize = false;
   int nClusters = 15;
   double threshold = 1e-3;

   DRWN_BEGIN_CMDLINE_PROCESSING (argc, argv)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
        DRWN_CMDLINE_INT_OPTION("-k", nClusters)
        DRWN_CMDLINE_REAL_OPTION("-t", threshold)
   DRWN_END_CMDLINE_PROCESSING(usage());

   if (DRWN_CMDLINE_ARGC != 2) {
       usage();
       return -1;
   }

   char * inputDir = DRWN_CMDLINE_ARGV[0]; 
   char * outputDir = DRWN_CMDLINE_ARGV[1];

   DRWN_ASSERT_MSG (drwnDirExists(inputDir), "image directory " << inputDir 
           << " does not exist");
   DRWN_ASSERT_MSG (drwnDirExists(outputDir), "image directory " << outputDir 
           << " does not exist");

   vector<string> baseNames = drwnDirectoryListing(inputDir, ".jpg", false, false);
   DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

   for (unsigned i = 0; i < baseNames.size(); i++) {
        string imgName = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS ("...processing image " << imgName);
        cout << "Processing image " << imgName << endl;

        // read given image
        string imgPath = string(inputDir) + DRWN_DIRSEP + imgName;
        cout << "Reading image: " << imgPath << endl;
        cv::Mat img = cv::imread (imgPath);
        cout << "Read image" << endl;
        
        // parameters
        const int H = img.rows;
        const int W = img.cols;

        // convert to needed format
        cv::Mat imgLab;
        cv::cvtColor (img, imgLab, CV_BGR2Lab);
        cout << "Create LAB image.." << endl;

        // slic algorithm invokation
        cv::Mat label(H, W, CV_8U, -1);
        slic (imgLab, label, nClusters, threshold);
        cout << "slic done.." << endl;

        // label out the boundaries of superpixels
        for (int y = 0; y < H; y ++) {
            for (int x = 0; x < W; x ++) {
                unsigned slabel = label.at<unsigned>(y, x);
                // cout << "("  << x << ", " << y << ")" << "slabel: " << slabel << endl;

                if (x - 1 < 0 || x + 1 > W || y - 1 < 0 || y + 1 > H) {
                    continue;
                }
                if (slabel != label.at<unsigned>(y, x+1) ||
                    slabel != label.at<unsigned>(y+1, x)) 
                {
                    img.at<cv::Vec3b>(y,x)[0] = 0;
                    img.at<cv::Vec3b>(y,x)[1] = 0;
                    img.at<cv::Vec3b>(y,x)[2] = 0;
                }
            }
        }
        cout << "slic received done.." << endl;

        for (int y = 0; y < H; y ++) {
            for (int x = 0; x < W; x ++) {
                unsigned slabel = label.at<unsigned>(y, x);
                // cout << "("  << x << ", " << y << ")" << "slabel: " << slabel << endl;

                img.at<cv::Vec3b>(y,x)[0] = 255/nClusters * slabel;
                img.at<cv::Vec3b>(y,x)[1] = 255/nClusters * slabel;
                img.at<cv::Vec3b>(y,x)[2] = 255/nClusters * slabel;
            }
        }

        /*
           if (bVisualize)
           {
           IplImage cvimg = (IplImage) img;
           IplImage *canvas = cvCloneImage(&cvimg);
           drwnShowDebuggingImage(canvas, "image", false);
           cvReleaseImage(&canvas);
           }
           */
        string outFilePath = string(outputDir) + DRWN_DIRSEP + baseNames[i] +
            "_seg.jpg";
        cout << "Write to file: " << outFilePath << endl;
        cv::imwrite (outFilePath, img);
        cout << "imwrite " << outFilePath << " done.." << endl;
   }
}
