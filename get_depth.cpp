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

using namespace cv;
using namespace std;

void blockMatching(Mat &disp, Mat imgL, Mat imgR){
  Mat grayL, grayR;

  cvtColor(imgL, grayL, CV_BGR2GRAY);
  cvtColor(imgR, grayR, CV_BGR2GRAY);

  Ptr<StereoBM> sbm = StereoBM::create(); 
  sbm->setBlockSize(17);
  sbm->setNumDisparities(32);

  sbm->compute(grayL, grayR, disp);
  normalize(disp, disp, 0, 255, CV_MINMAX, CV_8U);
}

void reconstruction(Mat img, Mat &Q, Mat &disp, Mat &rectified, Mat &XYZ){
  Mat mtxLeft = (Mat_<double>(3, 3) <<
    350.6847, 0, 332.4661,
    0, 350.0606, 163.7461,
    0, 0, 1);
  Mat distLeft = (Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  Mat mtxRight = (Mat_<double>(3, 3) <<
    351.9498, 0, 329.4456,
    0, 351.0426, 179.0179,
    0, 0, 1);
  Mat distRight = (Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  Mat R = (Mat_<double>(3, 3) <<
    0.9997, 0.0015, 0.0215,
    -0.0015, 1, -0.00008,
    -0.0215, 0.00004, 0.9997);
  //transpose(R, R);
  Mat T = (Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);

  Size stdSize = Size(640, 360);
  int width = img.cols;
  int height = img.rows;
  Mat imgL(img, Rect(0, 0, width/2, height));
  Mat imgR(img, Rect(width/2, 0, width/2, height));

  resize(imgL, imgL, stdSize);
  resize(imgR, imgR, stdSize);

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

  //imwrite("2_left.png", imgL);
  //imwrite("2_right.png", imgR);

  blockMatching(disp, imgL, imgR);

  // namedWindow("disp", WINDOW_NORMAL);
  // imshow("disp", disp);
  // waitKey(0);

  rectified = imgL;

  reprojectImageTo3D(disp, XYZ, Q);
  XYZ *= 0.001;
}

void getDepth(const string &imgPath) {
  auto imgSource = imread(imgPath);

  Mat Q, disp, rectified, XYZ, img;
  reconstruction(imgSource, Q, disp, rectified, XYZ);

  int index;
  string filename, savePath;

  index = imgPath.find_last_of('/');
  filename = imgPath.substr(index+1);
  savePath = imgPath.substr(0,index-7)+"/rectified/"+filename;
  imwrite(savePath, rectified);

  int index2 = filename.find_last_of('.');
  ifstream csvPath ( imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".csv" );
  string line, x, y, label;
  ofstream savefile;
  savePath = imgPath.substr(0,index-7)+"/results_3d/"+filename.substr(0,index2)+".csv";
  // cout << savePath << endl;
  savefile.open(savePath);
  while (getline(csvPath, line)) 
  {  
      stringstream liness(line);  
      getline(liness, x, ',');  
      getline(liness, y, ','); 
      getline(liness, label);
      
      Point position(stoi(x), stoi(y));
      Point3f point3D = XYZ.at<Point3f>(position);
      if(label == "1"){
        label = "blue";
        circle(rectified, position, 2, {255, 0, 0}, -1);
      }
      if(label == "0"){
        label = "yellow";
        circle(rectified, position, 2, {0, 255, 255}, -1);
      }
      if(label == "2"){
        label = "orange";
        circle(rectified, position, 2, {0, 0, 255}, -1);
      }
      
      // cout << position << " " << label << " " << point3D << endl;
      savefile << x+","+y+","+label+","+to_string(point3D.x)+","+to_string(point3D.y)+","+to_string(point3D.z)+"\n"; 
  }
  savefile.close();
  
  // namedWindow("probMapSoftmax", WINDOW_NORMAL);
  // imshow("probMapSoftmax", probMapSoftmax);
  // namedWindow("img", WINDOW_NORMAL);
  // imshow("img", rectified);
  // waitKey(0);

  // namedWindow("rectified", WINDOW_NORMAL);
  // imshow("rectified", rectified);
  // namedWindow("0", WINDOW_NORMAL);
  // imshow("0", probMapSplit[0]);
  // namedWindow("1", WINDOW_NORMAL);
  // imshow("1", probMapSplit[1]);
  // namedWindow("2", WINDOW_NORMAL);
  // imshow("2", probMapSplit[2]);
  // waitKey(0);

  savePath = imgPath.substr(0,index-7)+"/results_3d/"+filename;
  imwrite(savePath, rectified);
  // cout << savePath << endl;
}

void getAllImg(const string &imgFolderPath){
  boost::filesystem::path dpath(imgFolderPath);
  BOOST_FOREACH(const boost::filesystem::path& imgPath, make_pair(boost::filesystem::directory_iterator(dpath), boost::filesystem::directory_iterator())) {
    cout << imgPath.string() << endl;
	  getDepth(imgPath.string());
  }
}

int main(int argc, char **argv) {
  getAllImg(argv[1]);
}
