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
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include "opencv2/ximgproc/disparity_filter.hpp"

using namespace cv;
using namespace std;

void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create(); 
  sbmL->setBlockSize(13);
  sbmL->setNumDisparities(32);
  sbmL->compute(grayL, grayR, dispL);

  auto wls_filter = cv::ximgproc::createDisparityWLSFilter(sbmL);
  cv::Ptr<cv::StereoMatcher> sbmR = cv::ximgproc::createRightMatcher(sbmL);
  sbmR->compute(grayR, grayL, dispR);
  wls_filter->setLambda(8000);
  wls_filter->setSigmaColor(0.8);
  wls_filter->filter(dispL, imgL, disp, dispR);
  disp /= 16;

  cv::Mat disp8;
  cv::normalize(disp, disp8, 0, 255, 32, CV_8U);
  cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
  cv::imshow("disp", imgL+imgR);
  cv::waitKey(0);
}

void reconstruction(cv::Mat img, cv::Mat &Q, cv::Mat &disp, cv::Mat &rectified, cv::Mat &XYZ){
  // cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
  //   350.6847, 0, 332.4661,
  //   0, 350.0606, 163.7461,
  //   0, 0, 1);
  // cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  // cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
  //   351.9498, 0, 329.4456,
  //   0, 351.0426, 179.0179,
  //   0, 0, 1);
  // cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  // cv::Mat R = (cv::Mat_<double>(3, 3) <<
  //   0.9997, 0.0015, 0.0215,
  //   -0.0015, 1, -0.00008,
  //   -0.0215, 0.00004, 0.9997);
  // cv::Mat T = (cv::Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);
  // cv::Size stdSize = cv::Size(640, 360);

  //official
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  cv::Mat rodrigues = (cv::Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);
  cv::Mat R;
  cv::Rodrigues(rodrigues, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -0.12, 0, 0);
  cv::Size stdSize = cv::Size(672, 376);

  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  // cv::resize(imgL, imgL, stdSize);
  // cv::resize(imgR, imgR, stdSize);

  //std::cout << imgR.size() <<std::endl;

  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize, &validRoI[0], &validRoI[1]);

  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  // //check whether the camera is facing forward
  // cv::Mat rectify = imgL+imgR;
  // cv::line(rectify, cv::Point(336,0), cv::Point(336,378),cv::Scalar(0,0,0),1);
  // cv::namedWindow("rectified", cv::WINDOW_NORMAL);
  // cv::imshow("rectified", rectify);
  // cv::waitKey(0);

  // cv::imwrite("tmp/imgL.png", imgL);
  // cv::imwrite("tmp/imgR.png", imgR);
  // return;

  blockMatching(disp, imgL, imgR);

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", imgL+imgR);
  // cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void getDepth(const string &imgPath) {
  auto imgSource = imread(imgPath);

  Mat Q, disp, rectified, XYZ, img;
  reconstruction(imgSource, Q, disp, rectified, XYZ);

  int index, index2;
  string savePath;

  index = imgPath.find_last_of('.');
  savePath = imgPath.substr(0,index)+".csv";
  ifstream csvPath(savePath);
  cout << savePath << endl;

  string line, x, y, label;
  ofstream savefile;
  savefile.open(imgPath.substr(0,index)+"_3d.csv");
  while (getline(csvPath, line)) 
  {  
      stringstream liness(line);  
      getline(liness, x, ' ');  
      getline(liness, y, ' '); 
      getline(liness, label);
      
      Point position(stoi(x), stoi(y));
      Point3f point3D = XYZ.at<Point3f>(position);
      // cout << x+","+y+","+label+","+to_string(point3D.x)+","+to_string(point3D.y)+","+to_string(point3D.z)+"\n";
      savefile << x+","+y+","+label+","+to_string(point3D.x)+","+to_string(point3D.y)+","+to_string(point3D.z)+"\n"; 
  }
  savefile.close();
}

int main(int argc, char **argv) {
  // for(int i = 85; i <= 650; i++)
  //   getDepth("annotations/circle/results_circle_perfect/"+to_string(i)+".png");
  getDepth("annotations/circle/results_circle_perfect/100.png");
}
