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
*******************************************************************************/

#include "stdlib.h"
#include "math.h"
#include "cv.h"

#define TEST

/* Object nCluster keeping track of info of the cluster center */
class drwnCentroid {
    public:
        int l, a, b;
        int x, y;

        /* Update the drwnCentroid */
        void update (int l, int a, int b, int x, int y) {
            this->l = l;
            this->a = a;
            this->b = b;
            this->x = x;
            this->y = y;
        }
        /* Constructor */
        drwnCentroid () {};
        drwnCentroid (int l, int a, int b, int x, int y) {
            this->update (l, a, b, x, y);
        }
};

class drwnConnectedComponent {
    public:
        int id;  // id number of connected component
        int count;  // cumulative number of pixels in that connected component
        int acc_x;  // sum of horizental coordinate of all pixels within that component
        int acc_y;  // sum of vertical coordinate of all pixels within that component
        std::set< std::pair<int,int> > pixels;  // set of pixels 

        /* Constructor */
        drwnConnectedComponent(int id) {
            this->id = id;
            this->count = 0;
            this->acc_x = 0;
            this->acc_y = 0;
        }

        /* Add one pixel to this connected component */
        void add (int x, int y) {
            cout << "before add: " << (this->pixels).size()<< " after add: " << (this->pixels).size() <<<< endl;
            std::pair<int,int> point = std::make_pair(x,y);
            (this->pixels).insert (point);
            this->count += 1;
            cout <<  endl;
            this->acc_x += x;
            this->acc_y += y;
        }

        /* Incorporate all pixels in anotehr connected component */
        void merge (drwnConnectedComponent obj) {
            for (std::set< std::pair<int,int> >::iterator it = obj.pixels.begin();
                    it != obj.pixels.end(); ++it) {
                this->add((*it).first, (*it).second);
            }
        }

        /* Get the center of connected component */
        std::pair<int,int> getCenter() {
            cout << acc_x << ", " << count << ", " << acc_y << endl;
            return std::make_pair(acc_x/count, acc_y/count);
        }
};

bool connectedComponentCompare (drwnConnectedComponent cc1, drwnConnectedComponent cc2) {
    return cc1.count > cc2.count;
}

/* 
 * INPUT
 *   imgLab: image matrix in EILAB format
 *   nCluster: number of superpixel to generate
 *   label: matrix indicating the superpixel each pixel resides in, all -1 as
 *         input
 */
void slic (const cv::Mat imgLab, cv::Mat label, const int nCluster, double threshold) {
    DRWN_ASSERT (nCluster > 0);
    DRWN_ASSERT (threshold > 0.0);
    DRWN_ASSERT (1.0 > threshold);

    // height and width of the provided image
    const int H = imgLab.rows;
    const int W = imgLab.cols;
    DRWN_ASSERT (H > 0);
    DRWN_ASSERT (W > 0);

    // label must be initialized to be -1
    for (int y = 0; y < H; y ++) {
        for (int x = 0; x < W; x ++) {
            // cout << label.at<int>(y,x) << endl;
            DRWN_ASSERT (label.at<int>(y,x) == -1);
        }
    }

    // randomly pick up initial cluster center
    const int S = sqrt(H * W / nCluster);  // grid size
    const int gridPerRow = (W % S > 0)?(W / S + 1):(W / S);
    vector<drwnCentroid> ccs (nCluster, drwnCentroid());
    cout << "S :" << S << endl;
    cout << "gridPerRow: " << gridPerRow << endl;

    // randomize the position of drwnCentroid 
    /*{{{*/
    cout << "Randomly drwnCentroid selection begins.." << endl;
    for (int i = 0; i < nCluster; i ++) {
        int x, y;
        int l, a, b;
        int gridx = i % gridPerRow;
        int gridy = i / gridPerRow;
        x = (rand() % S) + gridx * S;
        y = (rand() % S) + gridy * S;
        // acquire lab color of the derived drwnCentroid
        l = imgLab.at<Vec3b>(y, x)[0];
        a = imgLab.at<Vec3b>(y, x)[1];
        b = imgLab.at<Vec3b>(y, x)[2];
        // update the drwnCentroid
        ccs[i].update(l, a, b, x, y);
        cout << "cluster[" << i << "]: (" << l << ", " << a << ", " << b << ", " << x << ", " << y << ")" << endl;
    }
    cout << "Randomly drwnCentroid selection ends.." << endl;
    /*}}}*/

    // Gradient computation
    /*{{{*/
    cout << "Gradient computation begins.." << endl;
    // Compute gradient magnitude of the given image
    cv::Mat gradient(H, W, CV_64F, cv::Scalar(0.0));
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int  countx = 0, county = 0;
            double gradientx = 0.0, gradienty = 0.0;
            int l = imgLab.at<Vec3b>(y, x)[0];
            int a = imgLab.at<Vec3b>(y, x)[1];
            int b = imgLab.at<Vec3b>(y, x)[2];
            if (x - 1 > 0) {
                int tmpl = imgLab.at<Vec3b>(y, x-1)[0];
                int tmpa = imgLab.at<Vec3b>(y, x-1)[1];
                int tmpb = imgLab.at<Vec3b>(y, x-1)[2];
                gradientx += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
                countx ++;
            }
            if (x + 1 < W) {
                int tmpl = imgLab.at<Vec3b>(y, x+1)[0];
                int tmpa = imgLab.at<Vec3b>(y, x+1)[1];
                int tmpb = imgLab.at<Vec3b>(y, x+1)[2];
                gradientx += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
                countx ++;
            }

            if (y - 1 > 0) {
                int tmpl = imgLab.at<Vec3b>(y-1, x)[0];
                int tmpa = imgLab.at<Vec3b>(y-1, x)[1];
                int tmpb = imgLab.at<Vec3b>(y-1, x)[2];
                gradienty += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
                county ++;
            }
            if (y + 1 < H) {
                int tmpl = imgLab.at<Vec3b>(y+1, x)[0];
                int tmpa = imgLab.at<Vec3b>(y+1, x)[1];
                int tmpb = imgLab.at<Vec3b>(y+1, x)[2];
                gradienty += abs(l-tmpl) + abs(a-tmpa) + abs(b-tmpb);
                county ++;
            }
            if (countx == 0 && county == 0) {
                gradient.at<double>(y, x) = INFINITY;
            } else {
                gradient.at<double>(y, x) = sqrt(pow(gradientx / countx, 2.0) +
                        pow(gradienty / county, 2.0));
            }
        }
    }
    // Move cluster center to the lowest gradient position
    int n = 5;
    for (int i = 0; i < nCluster; i ++) {
        int x = ccs[i].x;
        int y = ccs[i].y;
        int min_x = x;
        int min_y = y;
        double min_gradient = gradient.at<double>(y, x);
        for (int tmpy = std::max(y-n, 0); tmpy <= std::min(y+n, H-1); tmpy++) {
            for (int tmpx = std::max(x-n, 0); tmpx <= std::min(x+n, W-1); tmpx++) {
                double g = gradient.at<double>(tmpy, tmpx);
                if (g < min_gradient) {
                    min_x = tmpx;
                    min_y = tmpy;
                    min_gradient = g;
                }
            }
        }
        double min_l = imgLab.at<Vec3b>(min_y, min_x)[0];
        double min_a = imgLab.at<Vec3b>(min_y, min_x)[1];
        double min_b = imgLab.at<Vec3b>(min_y, min_x)[2];
        cout << "i: " << i 
             << " (" << ccs[i].x << ", " << ccs[i].y << ")" 
             << " (" << min_x << ", " << min_y << ", " << min_gradient << ")" 
             << endl;
        ccs[i].update (min_l, min_a, min_b, min_x, min_y);
    }

    cout << "Gradient computation ends.." << endl;
    /*}}}*/

    // Distance matrix represents the distance between one pixel and its
    // drwnCentroid
    cv::Mat distance(H, W, CV_64F, cv::Scalar(INFINITY));

    // iteration starts
    /*{{{*/
    int iter = 1;  // iteration number
    double m = 40.0; // relative importance between two type of distances
    cout << "Iteration starts.." << endl;
    while (true) {
        for (int i = 0; i < nCluster; i ++) {
            // acquire labxy attribute of drwnCentroid
            int x = ccs[i].x;
            int y = ccs[i].y;
            int l = ccs[i].l;
            int a = ccs[i].a;
            int b = ccs[i].b;

            // look around its 2S x 2S region
            for (int tmpy = std::max(y-S, 0); tmpy <= std::min(y+S, H-1); tmpy++) {
                for (int tmpx = std::max(x-S, 0); tmpx <= std::min(x+S, W-1); tmpx++) {
                    int tmpl = imgLab.at<Vec3b>(tmpy, tmpx)[0];
                    int tmpa = imgLab.at<Vec3b>(tmpy, tmpx)[1];
                    int tmpb = imgLab.at<Vec3b>(tmpy, tmpx)[2];

                    double color_distance = sqrt (pow(tmpl - l, 2.0) +
                            pow(tmpa - a, 2.0) + pow(tmpb - b, 2.0)); 
                    double spatial_distance = sqrt (pow(tmpx - x, 2.0) +
                            pow(tmpy - y, 2.0));
                    double D = sqrt (pow(color_distance, 2.0) +
                            pow(spatial_distance * m / S, 2.0));

                    // if distance is smaller, update the drwnCentroid it belongs to
                    if (D < distance.at<double>(tmpy, tmpx)) {
                        distance.at<double>(tmpy, tmpx) = D;
                        label.at<int>(tmpy, tmpx) = i;
                    }
                }
            }
        }

        // Compute new cluster center by taking the mean of each dimension
        vector<unsigned int> count = vector<unsigned int> (nCluster, 0.0);
        vector<double> sumx = vector<double> (nCluster, 0.0);
        vector<double> sumy = vector<double> (nCluster, 0.0);
        vector<double> suml = vector<double> (nCluster, 0.0);
        vector<double> suma = vector<double> (nCluster, 0.0);
        vector<double> sumb = vector<double> (nCluster, 0.0);

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int clusterIdx = label.at<int>(y, x);
                if (clusterIdx < 0) {
                    // cout << x << "," << y << "clusterindx: " << clusterIdx << endl;
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

        // statistics about each superpixel
        cout << "At iteration " << iter << ":" << endl;
        for (int i = 0; i < nCluster; i ++) {
            cout << "i = " << i << ", count = " << count[i] << endl;
        }

        vector<drwnCentroid> newccs (nCluster, drwnCentroid());
        for (int i = 0; i < nCluster; i ++) {
            int x = (int) (1.0 * sumx[i] / count[i]);
            int y = (int) (1.0 * sumy[i] / count[i]);
            int l = (int) (1.0 * suml[i] / count[i]);
            int a = (int) (1.0 * suma[i] / count[i]);
            int b = (int) (1.0 * sumb[i] / count[i]);
            newccs[i].update(l, a, b, x, y);
        }

        // Compute residual error E using L-2 norm
        double E = 0.0;
        for (int i = 0; i < nCluster; i ++) {
            double error = 0.0;
            error += pow((ccs[i].x - newccs[i].x), 2.0);
            error += pow((ccs[i].y - newccs[i].y), 2.0);
            error += pow((ccs[i].l - newccs[i].l), 2.0);
            error += pow((ccs[i].a - newccs[i].a), 2.0);
            error += pow((ccs[i].b - newccs[i].b), 2.0);
            E += error;
        }
        E /= nCluster;
        
        // reassign new to previous
        for (int i = 0; i < nCluster; i ++) {
            ccs[i] = newccs[i];
        }

        cout << "Iteration: " << iter++ << ", error: " << E << endl;
        // Stop iteration until specified precision is reached
        if (E < threshold) {
            break;
        }
    }
/*}}}*/

    // post processing: enforce 4-connectivity
    cv::Mat cclabel (H, W, CV_32S, -1);
    int ccCount = 0;
    std::vector<drwnConnectedComponent> cclist;
    // find all connected component
    cout << "Find all connected component starts.." << endl; 
    for (int y = 0; y < H; y ++) {
        for (int x = 0; x < W; x ++) {
            int west_label = x-1>=0?label.at<int>(y,x-1):-1;
            int north_label = y-1>=0?label.at<int>(y-1,x):-1;
            int current_label = label.at<int>(y,x);
            bool west_equal = x > 0 && current_label == west_label;
            bool north_equal = y > 0 && current_label == north_label;
            if (west_equal && north_equal) {
                // cout << "equal" << endl;
                if (cclist[north_label].id == cclist[west_label].id) {
                    // west and north has already been in the same connected component
                    cclist[north_label].add(x, y);
                } else {
                    // merge west and north connected component and add to it
                    cclist[north_label].merge(cclist[west_label]);
                    ccCount--;
                }
                continue;
            }
            if (west_equal) {
                // cout << "west_equal" << endl;
                // add to west connected component
                cclabel.at<int>(y,x) = west_label;
                cclist[west_label].add(x, y);
                continue;
            }
            if (north_equal) {
                //cout << "north_equal" << endl;
                // add to north connected component
                cclabel.at<int>(y,x) = north_label;
                cclist[north_label].add(x, y);
                continue;
            }
            // new connected component
            // cout << "sdf" << endl;
            ccCount += 1;
            cclabel.at<int>(y,x) = ccCount;
            drwnConnectedComponent* newcc = new drwnConnectedComponent(ccCount);
            cclist.push_back(*newcc);
        }
    }
    cout << "Componen number: " << ccCount << endl;
    cout << "Find all connected component ends.." << endl;
    // sort all connected components with its magnitude
    std::list<drwnConnectedComponent> sortlist (cclist.begin(), cclist.end());
    sortlist.sort (connectedComponentCompare);
    cclist = std::vector<drwnConnectedComponent>(sortlist.begin(), sortlist.end());
    // merge smallest component to its nearest connected component
    // DRWN_ASSERT (nCluster > cclist.size());
    for (int i = nCluster; i < cclist.size(); i ++) {
        if (cclist[i].count <= 0) continue;
        cout << "size of cc [" << i << "]: " << cclist[i].count << endl;
        double min_dist = INFINITY;
        int closest_cc = -1;
        for (int j = 0; j < nCluster; j ++) {
            std::pair<int,int> pi = cclist[i].getCenter();
            std::pair<int,int> pj = cclist[j].getCenter();
            double tmp_dist = sqrt(pow((pi.first - pj.first),2) + pow((pi.second - pj.second),2));
            if (tmp_dist < min_dist) {
                min_dist = tmp_dist;
                closest_cc = j;
            }
        }
        cclist[closest_cc].merge(cclist[i]);
        //delete cclist[i];
    }
    cout << "Merge all connected component ends.." << endl;

    // mark up in the label matrix
    for (int i = 0; i < nCluster; i ++) {
        for (std::set< std::pair<int,int> >::iterator it = cclist[i].pixels.begin();
                it != cclist[i].pixels.end(); ++it) {
            label.at<int>((*it).second, (*it).first) = i;
        }
    }

}
