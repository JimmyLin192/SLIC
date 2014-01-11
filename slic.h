/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    slic.h
** AUTHOR(S):   Jimmy Lin <JimmyLin@utexas.edu>
** DESCRIPTION:
**   Darwin GUI for creating and evaluating data flow algorithms.
**
*****************************************************************************/

#include "stdlib.h"
#include "math.h"
#include "cv.h"

#define TEST

/* Object keeping track of info of the cluster center */
class centroid {
    public:
        int x, y;
        double l, a, b;

        /* Update the centroid */
        void update (double l, double a, double b, int x, int y) {
            this->l = l;
            this->a = a;
            this->b = b;
            this->x = x;
            this->y = y;
        }
        /* Constructor */
        centroid () {};
        centroid (double l, double a, double b, int x, int y) {
            this->update (l, a, b, x, y);
        }
};

/* 
 * INPUT
 *   imgLab: image matrix in EILAB format
 *   k: number of superpixel to generate
 * RETURN
 *   label: matrix indicating the superpixel each pixel resides in
 */
cv::Mat slic (cv::Mat imgLab, const int k, double threshold) {
    if (k < 0) {
        cout << "invalid cluster number specification. " << endl;
    }

    // height and width of the provided image
    const int H = imgLab.rows;
    const int W = imgLab.cols;

    // randomly pick up initial cluster center
    const int S = sqrt(H * W / k);  // grid size
    const int gridPerRow = W / S;
    vector<centroid> ccs (k, centroid());

    for (int i = 0; i < k; i ++) {
        // randomize the position of centroid
        int x, y;
        double l, a, b;
        int gridx = i % gridPerRow;
        int gridy = i / gridPerRow;
        x = (rand() % S) + gridx * S;
        y = (rand() % S) + gridy * S;
        // acquire lab color of the derived centroid
        l = imgLab.at<Vec3b>(y, x)[0];
        a = imgLab.at<Vec3b>(y, x)[1];
        b = imgLab.at<Vec3b>(y, x)[2];
        ccs[i].update(l, a, b, x, y);
    }

    cout << "Randomly pick up centroid.." << endl;

    // Compute gradient magnitude of the given image
    cv::Mat gradient = cv::Mat (H, W, CV_64F, 0.0);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double gradientx = 0.0, gradienty = 0.0;
            double l = imgLab.at<Vec3b>(y, x)[0];
            double a = imgLab.at<Vec3b>(y, x)[1];
            double b = imgLab.at<Vec3b>(y, x)[2];
            if (x - 1 > 0) {
                double tmpl = imgLab.at<Vec3b>(y, x-1)[0];
                double tmpa = imgLab.at<Vec3b>(y, x-1)[1];
                double tmpb = imgLab.at<Vec3b>(y, x-1)[2];
                gradientx += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
            } else gradientx = INFINITY;

            if (x + 1 < W) {
                double tmpl = imgLab.at<Vec3b>(y, x+1)[0];
                double tmpa = imgLab.at<Vec3b>(y, x+1)[1];
                double tmpb = imgLab.at<Vec3b>(y, x+1)[2];
                gradientx += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
            } else gradientx = INFINITY;

            if (y - 1 > 0) {
                double tmpl = imgLab.at<Vec3b>(y-1, x)[0];
                double tmpa = imgLab.at<Vec3b>(y-1, x)[1];
                double tmpb = imgLab.at<Vec3b>(y-1, x)[2];
                gradienty += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
            } else gradienty = INFINITY;

            if (y + 1 < H) {
                double tmpl = imgLab.at<Vec3b>(y-1, x)[0];
                double tmpa = imgLab.at<Vec3b>(y-1, x)[1];
                double tmpb = imgLab.at<Vec3b>(y-1, x)[2];
                gradienty += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
            } else gradienty = INFINITY;

            if (gradientx == INFINITY || gradienty == INFINITY)
                gradient.at<double>(y, x) = INFINITY;
            else
                gradient.at<double>(y, x) = sqrt(pow(gradientx, 2.0) + pow(gradienty, 2.0));
        }
    }
    /*
    // Move cluster center to the lowest gradient position
    int n = 1;
    for (int i = 0; i < k; i ++) 
    {
        double min_gradient = INFINITY;
        int min_x = -1;
        int min_y = -1;
        for (int y = std::max(y-n, 0); y < std::min(y+n, H); y++)
        {
            for (int x = std::max(x-n, 0); x < std::min(x+n, W); x++)
            {
                double g = gradient.at<double>(y, x);
                if (g < min_gradient) {
                    min_x = x;
                    min_y = y;
                    min_gradient = g;
                }
            }
        }
        double min_l = imgLab.at<Vec3b>(min_y, min_x)[0];
        double min_a = imgLab.at<Vec3b>(min_y, min_x)[1];
        double min_b = imgLab.at<Vec3b>(min_y, min_x)[2];
        cout << "i: " << i 
             << " (" << ccs[i].x << "," << ccs[i].y << ")" 
             << " (" << min_x << "," << min_y << ")" 
             << endl;
        update(ccs[i], min_x, min_y, min_l, min_a, min_b);
    }
    */

    cout << "Move the centroid to the lowest gradient position" << endl;
    
    // label matrix indicate the cluster number one pixel is in
    cv::Mat label = cv::Mat (H, W, CV_8U, -1);
    // distance matrix represents the distance between one pixel and its
    // centroid
    cv::Mat distance = cv::Mat (H, W, CV_64F, INFINITY);

    int iter = 1;
    while (true) {
        for (int i = 0; i < k; i ++) {
            // acquire labxy attribute of centroid
            int x = ccs[i].x;
            int y = ccs[i].y;
            double l = ccs[i].l;
            double a = ccs[i].a;
            double b = ccs[i].b;

            // look around its 2S x 2S region
            for (int tmpy = std::max(y-S, 0); tmpy < std::min(y+S, H); tmpy++) 
            {
                for (int tmpx = std::max(x-S, 0); tmpx < std::min(x+S, W); tmpx++) {
                    double tmpl = imgLab.at<Vec3b>(tmpy, tmpx)[0];
                    double tmpa = imgLab.at<Vec3b>(tmpy, tmpx)[1];
                    double tmpb = imgLab.at<Vec3b>(tmpy, tmpx)[2];

                    double color_distance = sqrt (pow(tmpl - l, 2.0) +
                            pow(tmpa - a, 2.0) + pow(tmpb - b, 2.0)); 
                    double spatial_distance = sqrt (pow(tmpx - x, 2.0) +
                            pow(tmpy - y, 2.0));

                    // FIXME: to be refined formula for ultimate distance
                    double D = sqrt (pow(color_distance, 2.0) +
                            pow(spatial_distance / S, 2.0));

                    // distance is smaller, update the centroid it belong to
                    if (D < distance.at<double>(tmpy, tmpx)) {
                        // cout << "comparison: (" << D << ", " << distance.at<double>(tmpy, tmpx) <<")" << endl;
                        distance.at<double>(tmpy, tmpx) = D;
                        label.at<unsigned>(tmpy, tmpx) = i;
                    }
                }

            }
        }

        // Compute new cluster center by taking the mean of each dimension
        vector<unsigned> count = vector<unsigned> (k, 0);
        vector<double> sumx = vector<double> (k, 0.0);
        vector<double> sumy = vector<double> (k, 0.0);
        vector<double> suml = vector<double> (k, 0.0);
        vector<double> suma = vector<double> (k, 0.0);
        vector<double> sumb = vector<double> (k, 0.0);

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                unsigned clusterIdx = label.at<unsigned>(y, x);
                //cout << "clusterIdx: " << clusterIdx << endl;
                if (clusterIdx > (unsigned) k) {
                    continue;
                }
                count[clusterIdx] ++;
                sumx[clusterIdx] += x;
                sumy[clusterIdx] += y;
                suml[clusterIdx] += imgLab.at<Vec3b>(y,x)[0];
                suma[clusterIdx] += imgLab.at<Vec3b>(y,x)[1];
                sumb[clusterIdx] += imgLab.at<Vec3b>(y,x)[2];
            }
        }

        /*
        // statistics about each superpixel
        for (int i = 0; i < k; i ++)
        {
            cout << "i = " << i << ", count = " << count[i] << endl;
        }
        */

        vector<centroid> newccs (k, centroid());
        for (int i = 0; i < k; i ++) {
            int x = (int) (sumx[i] / count[i]);
            int y = (int) (sumy[i] / count[i]);
            double l = (int) (suml[i] / count[i]);
            double a = (int) (suma[i] / count[i]);
            double b = (int) (sumb[i] / count[i]);
            newccs[i].update( x, y, l, a, b);
        }

        // Compute residual error E using L-2 norm
        double E = 0.0;
        for (int i = 0; i < k; i ++) {
            double error = 0.0;
            error += pow((ccs[i].x - newccs[i].x), 2.0);
            error += pow((ccs[i].y - newccs[i].y), 2.0);
            error += pow((ccs[i].l - newccs[i].l), 2.0);
            error += pow((ccs[i].a - newccs[i].a), 2.0);
            error += pow((ccs[i].b - newccs[i].b), 2.0);
            E += error;
        }
        E /= k;
        
        // reassign new to previous
        for (int i = 0; i < k; i ++) {
            ccs[i] = newccs[i];
        }

        cout << "Iteration: " << (iter++) << ", error: " << E << endl;
        // Stop iteration until specified precision is reached
        if (E < threshold) {
            break;
        }
    }

    return label;
}
