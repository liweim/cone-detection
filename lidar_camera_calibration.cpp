/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/ximgproc/disparity_filter.hpp"

using namespace cv;
using namespace std;

void blockMatching(Mat &disp, Mat imgL, Mat imgR){
  Mat grayL, grayR, dispL, dispR;

  cvtColor(imgL, grayL, 6);
  cvtColor(imgR, grayR, 6);

  Ptr<StereoBM> sbmL = StereoBM::create(); 
  sbmL->setBlockSize(13);
  sbmL->setNumDisparities(32);
  sbmL->compute(grayL, grayR, dispL);

  auto wls_filter = ximgproc::createDisparityWLSFilter(sbmL);
  Ptr<StereoMatcher> sbmR = ximgproc::createRightMatcher(sbmL);
  sbmR->compute(grayR, grayL, dispR);
  wls_filter->setLambda(8000);
  wls_filter->setSigmaColor(0.8);
  wls_filter->filter(dispL, imgL, disp, dispR);

  normalize(disp, disp, 0, 255, 32, CV_8U);
  // namedWindow("disp", WINDOW_AUTOSIZE);
  // imshow("disp", disp);
  // waitKey(10);
}

void reconstruction(const string imgPath, Mat &Q, Mat &disp, Mat &rectified, Mat &XYZ){  
  // Mat mtxLeft = (Mat_<double>(3, 3) <<
  //   350.6847, 0, 332.4661,
  //   0, 350.0606, 163.7461,
  //   0, 0, 1);
  // Mat distLeft = (Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  // Mat mtxRight = (Mat_<double>(3, 3) <<
  //   351.9498, 0, 329.4456,
  //   0, 351.0426, 179.0179,
  //   0, 0, 1);
  // Mat distRight = (Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  // Mat R = (Mat_<double>(3, 3) <<
  //   0.9997, 0.0015, 0.0215,
  //   -0.0015, 1, -0.00008,
  //   -0.0215, 0.00004, 0.9997);
  // Mat T = (Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);
  // Size stdSize = Size(640, 360);

  //official
  Mat mtxLeft = (Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  Mat distLeft = (Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  Mat mtxRight = (Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  Mat distRight = (Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  Mat rodrigues = (Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);
  Mat R;
  Rodrigues(rodrigues, R);
  Mat T = (Mat_<double>(3, 1) << -0.12, 0, 0);
  Size stdSize = Size(672, 376);

  auto img = imread(imgPath);
  int width = img.cols;
  int height = img.rows;
  Mat imgL(img, Rect(0, 0, width/2, height));
  Mat imgR(img, Rect(width/2, 0, width/2, height));

  imgL.copyTo(rectified);

  // resize(imgL, imgL, stdSize);
  // resize(imgR, imgR, stdSize);

  //cout << imgR.size() <<endl;

  Mat R1, R2, P1, P2;
  Rect validRoI[2];
  stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 0.0, stdSize, &validRoI[0], &validRoI[1]);

  Mat rmap[2][2];
  initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  remap(imgL, imgL, rmap[0][0], rmap[0][1], INTER_LINEAR);
  remap(imgR, imgR, rmap[1][0], rmap[1][1], INTER_LINEAR);


  blockMatching(disp, imgL, imgR);

  int index;
  string filename, savePath;
  index = imgPath.find_last_of('/');
  filename = imgPath.substr(index+1);
  savePath = imgPath.substr(0,index-7)+"/disp/"+filename;
  imwrite(savePath, disp);

  // savePath = imgPath.substr(0,index-7)+"/right/"+filename;
  // imwrite(savePath, imgR);

  // namedWindow("after", WINDOW_NORMAL);
  // imshow("after", imgL);
  // waitKey(0);

  reprojectImageTo3D(disp, XYZ, Q);
}

void xyz2xy(Mat Q, Point3f xyz, Point2f &xy){
  double X = xyz.x;
  double Y = xyz.y;
  double Z = xyz.z;
  double Cx = -Q.at<double>(0,3);
  double Cy = -Q.at<double>(1,3);
  double f = Q.at<double>(2,3);
  double a = Q.at<double>(3,2);
  double b = Q.at<double>(3,3);
  double d = (f - Z * b ) / ( Z * a);
  xy.x = X * ( d * a + b ) + Cx;
  xy.y = Y * ( d * a + b ) + Cy;
}

float_t depth2resizeRate(double x, double y){
  return 2*(1.6078-0.4785*sqrt(sqrt(x*x+y*y)));
}

void calibration(int id){
  //Given RoI in 3D world, project back to the camera frame and then detect
  double yShift = 1.872;
  double zShift = 0.833;
  double theta = 0*3.1415/180;

  string imgPath("annotations/circle/images/"+to_string(id+8)+".png");
  Mat disp, Q, rectified, XYZ;
  reconstruction(imgPath, Q, disp, rectified, XYZ);

  ifstream csvPath("annotations/circle/lidarResultsTogether/"+to_string(id)+".csv");
  string line, x, y, z; 
  double X, Y, Z;
  vector<Point3d> pts;

  while (getline(csvPath, line)) 
  {  
    stringstream liness(line);  
    getline(liness, x, ',');
    getline(liness, y, ',');
    getline(liness, z, ',');
    X = stod(x);
    Y = zShift+stod(z);
    Z = yShift+stod(y);
    pts.push_back(Point3d(X*cos(theta)+Z*sin(theta), 1, -X*sin(theta)+Z*cos(theta)));
  }

  for(size_t i = 0; i < pts.size(); i++){
    Point2f point2D;
    xyz2xy(Q, pts[i], point2D);

    int x = point2D.x;
    int y = point2D.y;
    // cout << x << " " << y << endl;

    // cout << "Camera region center: " << x << ", " << y << endl;
    float_t ratio = depth2resizeRate(pts[i].x, pts[i].z);
    if (ratio > 0) {
      int length = ratio * 25;
      int radius = (length-1)/2;
      // cout << "radius: " << radius << endl;

      Rect roi;
      roi.x = max(x - radius, 0);
      roi.y = max(y - radius, 0);
      roi.width = min(x + radius, rectified.cols) - roi.x;
      roi.height = min(y + radius, rectified.rows) - roi.y;

      //circle(img, Point (x,y), radius, Scalar (0,0,0));
      // // circle(disp, Point (x,y), 3, 0, CV_FILLED);
      //namedWindow("roi", WINDOW_NORMAL);
      //imshow("roi", img);
      //waitKey(0);
      //destroyAllWindows();
      if (0 > roi.x || 0 > roi.width || roi.x + roi.width > rectified.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > rectified.rows){
        cout << "Wrong roi!" << endl;
      }
      else{
        circle(rectified, Point (x,y), radius, Scalar(0,0,255), 2);
      }
    }
  }

  namedWindow("disp", WINDOW_NORMAL);
  imshow("disp", rectified);
  waitKey(0);
}

int main(int argc, char **argv) {
  for(int i = 80; i < 630; i++){
    cout << i << endl;
    calibration(i);
  }
}
