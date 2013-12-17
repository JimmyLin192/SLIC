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

   DRWN_BEGIN_CMDLINE_PROCESSING (argc, argv);
   DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize);
   DRWN_END_CMDLINE_PROCESSING(usage());


   if (DRWN_CMDLINE_ARGC != 2) {
       usage();
       return -1;
   }

   char * inputDir = DRWN_CMDLINE_ARGV[0]; 
   char * outputDir = DRWN_CMDLINE_ARGV[1];

   DRWN_ASSERT_MSG (drwnDirExists(imgDir), "image directory " << imgDir 
           << " does not exist");

   vector<string> baseNames = drwnDirectoryListing(imgDir, ".jpg", false, false);
   DRWN_LOG_MESSAGE("Loading " << baseNames.size() << " images and labels...");

   drwnClassifierDataset dataset;

   for (unsigned i = 0; i < baseNames.size(); i++) 
   {
        String imgName = baseNames[i] + ".jpg";
        DRWN_LOG_STATUS ("...processing image " << baseNames[i]);

        // read given image
        cv::Mat img = cv::imread(string(inputDir) + DRWN_DIRSEP + imgName);
        // convert to needed format
        cv::Mat imgLab;
        cv::cvtColor(imgRgb, imgLab, cv::CV_BGR2Lab);

        slic ();
        double b = imgLab.at<double>(y, x)[2];

        if (bVisualize) 
        {


        }


   }
}
