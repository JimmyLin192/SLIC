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

#include "slic.h"

using namespace std;
using namespace Eigen;

// HELP FUNCTION
void usage() {
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./slic [OPTIONS] <inputDir> <outputDir>\n";
    cerr << "OPTIONS:\n"
        // << "  -o <model>        :: output model\n"
        << "  -x                :: visualize\n"
        << DRWN_STANDARD_OPTIONS_USAGE
        << endl;
}

// MAIN FUNCTION
int main (int argc, char * argv [])
{
   bool bVisualize = false;

   DRWN_BEGIN_CMDLINE_PROCESSING (argc, argv)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
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

   //drwnClassifierDataset dataset;

   for (unsigned i = 0; i < baseNames.size(); i++) 
   {
        string imgName = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS ("...processing image " << baseNames[i]);

        // read given image
        cv::Mat img = cv::imread(string(inputDir) + DRWN_DIRSEP + imgName);
        // resulted image
        cv::Mat imgseg = cv::Mat(img);
        // parameters
        const int H = img.rows;
        const int W = img.cols;
        // convert to needed format
        cv::Mat imgLab;
        cv::cvtColor(img, imgLab, CV_BGR2Lab);

        cv::Mat label = slic (imgLab, 50, 1);
        for (int y = 0; y < H; y ++) 
        {
            for (int x = 0; x < W; x ++)
            {
                unsigned slabel = label.at<unsigned>(y, x);
                if (x - 1 < 0 || x + 1 > W || y - 1 < 0 || y + 1 > H)
                    continue;
                if (slabel != label.at<unsigned>(y, x-1) ||
                    slabel != label.at<unsigned>(y, x+1) ||
                    slabel != label.at<unsigned>(y-1, x) ||
                    slabel != label.at<unsigned>(y+1, x) ) 
                {
                    imgseg.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        /*
        if (bVisualize)
        {
            IplImage cvimg = (IplImage) img;
            IplImage *canvas = cvCloneImage(&cvimg);
            drwnShowDebuggingImage(canvas, "image", false);
            cvReleaseImage(&canvas);
            IplImage cvimgseg = (IplImage) imgseg;
            IplImage *canvasseg = cvCloneImage(&cvimgseg);
            drwnShowDebuggingImage(canvasseg, "image", false);
            cvReleaseImage(&canvasseg);
        }
        */
        cv::imwrite(outputDir+imgName, imgseg);

   }
}
