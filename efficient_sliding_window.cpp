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

#include "tiny_dnn/tiny_dnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

tiny_dnn::network<tiny_dnn::sequential> nn;

void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR;

  cv::cvtColor(imgL, grayL, CV_BGR2GRAY);
  cv::cvtColor(imgR, grayR, CV_BGR2GRAY);

  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(); 
  sbm->setBlockSize(17);
  sbm->setNumDisparities(32);

  sbm->compute(grayL, grayR, disp);
  cv::normalize(disp, disp, 0, 255, CV_MINMAX, CV_8U);
}

void reconstruction(cv::Mat img, cv::Mat &Q, cv::Mat &disp, cv::Mat &rectified, cv::Mat &XYZ){
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    350.6847, 0, 332.4661,
    0, 350.0606, 163.7461,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    351.9498, 0, 329.4456,
    0, 351.0426, 179.0179,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  cv::Mat R = (cv::Mat_<double>(3, 3) <<
    0.9997, 0.0015, 0.0215,
    -0.0015, 1, -0.00008,
    -0.0215, 0.00004, 0.9997);
  //cv::transpose(R, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);

  cv::Size stdSize = cv::Size(640, 360);
  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  cv::resize(imgL, imgL, stdSize);
  cv::resize(imgR, imgR, stdSize);

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

  //cv::imwrite("2_left.png", imgL);
  //cv::imwrite("2_right.png", imgR);

  blockMatching(disp, imgL, imgR);

  cv::namedWindow("disp", cv::WINDOW_NORMAL);
  cv::imshow("disp", disp);
  cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
  XYZ *= 0.001;
}

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convertImage(cv::Mat img,
                   int w,
                   int h,
                   tiny_dnn::vec_t& data){

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        data[c * w * h + y * w + x] =
          resized.at<cv::Vec3b>(y, x)[c] / 255.0;
      }
    }
  }
}

void constructNetwork(const std::string &dictionary, int width, int height) {
  using conv    = tiny_dnn::convolutional_layer;
  using relu    = tiny_dnn::relu_layer;
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::backend_t::avx;

  nn << conv(width, height, 7, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-6, height-6, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-12, height-12, 5, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-16, height-16, 5, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-20, height-20, 3, 32, 64, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-22, height-22, 3, 64, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type);

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;
}

void softmax(cv::Vec4d x, cv::Vec4d &y) {
  double min, max, denominator = 0;
  cv::minMaxLoc(x, &min, &max);
  for (int j = 0; j < 4; j++) {
    y[j] = std::exp(x[j] - max);
    denominator += y[j];
  }
  for (int j = 0; j < 4; j++) {
    y[j] /= denominator;
  }
}


std::vector <cv::Point> imRegionalMax(cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
    cv::Mat scratch = input.clone();
    // std::cout<<scratch<<std::endl;
    // cv::GaussianBlur(scratch, scratch, cv::Size(3,3), 0, 0);
    std::vector <cv::Point> locations(0);
    locations.reserve(nLocMax); // Reserve place for fast access
    for (int i = 0; i < nLocMax; i++) {
        cv::Point location;
        double maxVal;
        cv::minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
        if (maxVal > threshold) {
            int row = location.y;
            int col = location.x;
            locations.push_back(cv::Point(col, row));
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<double>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
    return locations;
}

// void detectRoI(const std::string &modelPath, const std::string &imgPath, double threshold) {
//   int patchSize = 25;
//   int patchRadius = (patchSize-1)/2;
//   double factor = 2;
//   int inputSize = patchSize * factor;
//   int inputRadius = (inputSize-1)/2;
//   int outputSize  = inputSize - (patchSize - 1);

//   tiny_dnn::network<tiny_dnn::sequential> nn;

//   constructNetwork(nn, tiny_dnn::core::default_engine(), inputSize, inputSize);

//   // load nets
//   std::ifstream ifs(modelPath.c_str());
//   ifs >> nn;

//   // convert imagefile to vec_t
//   // tiny_dnn::image<> img(imgPath, tiny_dnn::image_type::bgr);

//   // manual roi
//   // (453, 237, 0.96875, "orange")
//   // (585, 211,	0.625,	"orange");
//   // (343, 185,	0.25,	"yellow");
//   // (521, 198,	0.375,	"yellow");
//   // (634,	202,	0.375,	"yellow");
//   // (625,	191,	0.34375,	"blue");
//   // (396,	183,	0.34375,	"blue");

//   int cx = 396;
//   int cy = 183;
//   double ratio = 0.34375;

//   int roiSize = factor * ratio * patchSize;
//   int roiRadius = (roiSize-1)/2;
//   cv::Rect roi;
//   roi.x = std::max(cx - roiRadius, 0);
//   roi.y = std::max(cy - roiRadius, 0);
//   roi.width = std::min(cx + roiRadius, 640) - roi.x;
//   roi.height = std::min(cy + roiRadius, 360) - roi.y;
//   auto img = cv::imread(imgPath);
//   auto patchImg = img(roi);
//   cv::resize(patchImg, patchImg, cv::Size(inputSize, inputSize));

//   cv::namedWindow("img", cv::WINDOW_NORMAL);
//   cv::imshow("img", patchImg);
//   cv::waitKey(0);
//   cv::destroyAllWindows();

//   // convert imagefile to vec_t
//   tiny_dnn::vec_t data;
//   convertImage(patchImg, inputSize, inputSize, data);

//   // recognize
//   auto prob = nn.predict(data);

//   cv::Mat probMap = cv::Mat::zeros(outputSize, outputSize, CV_32FC4);
//   for (size_t c = 0; c < 4; ++c)
//     for (size_t y = 0; y < outputSize; ++y)
//       for (size_t x = 0; x < outputSize; ++x)
//          probMap.at<cv::Vec4f>(y, x)[c] = prob[c * outputSize * outputSize + y * outputSize + x];

//   cv::Vec4f probSoftmax(4);
//   cv::Mat probMapSoftmax = cv::Mat::zeros(outputSize, outputSize, CV_32FC3);
//   for (size_t y = 0; y < outputSize; ++y)
//     for (size_t x = 0; x < outputSize; ++x){
//       softmax(probMap.at<cv::Vec4f>(y, x), probSoftmax);
//       for (size_t c = 0; c < 3; ++c)
//         if(probSoftmax[c+1] > threshold)
//           probMapSoftmax.at<cv::Vec3f>(y, x)[c] = probSoftmax[c+1];
//     }
//   cv::Mat probMapSplit[3];
//   cv::split(probMapSoftmax, probMapSplit);

//   for (size_t c = 0; c < 3; ++c){
//     cv::namedWindow("img", cv::WINDOW_NORMAL);
//     cv::imshow("img", probMapSplit[c]);
//     cv::waitKey(0);
//     cv::destroyAllWindows();
//   }

//   std::vector <cv::Point> yellow, blue, orange;

//   int nLocMax = 1;
//   int minDistBtwLocMax = 10;
//   yellow = imRegionalMax(probMapSplit[0], nLocMax, threshold, minDistBtwLocMax);
//   blue = imRegionalMax(probMapSplit[1], nLocMax, threshold, minDistBtwLocMax);
//   orange = imRegionalMax(probMapSplit[2], nLocMax, threshold, minDistBtwLocMax);

//   cv::Point position, positionShift = cv::Point(patchRadius*ratio + cx - roiRadius, patchRadius*ratio + cy - roiRadius);

//   if (yellow.size()>0){
//     for(int i=0; i<yellow.size(); i++){
//       position = yellow[i]*ratio + positionShift;
//       cv::circle(img, position, 1, {0, 255, 255}, -1);
//     }
//   }
//   if (blue.size()>0){
//     for(int i=0; i<blue.size(); i++){
//       position = blue[i]*ratio + positionShift;
//       cv::circle(img, position, 1, {255, 0, 0}, -1);
//     }
//   }
//   if (orange.size()>0){
//     for(int i=0; i<orange.size(); i++){
//       position = orange[i]*ratio + positionShift;
//       cv::circle(img, position, 1, {0, 165, 255}, -1);
//     }
//   }
//   cv::namedWindow("img", cv::WINDOW_NORMAL);
//   cv::imshow("img", img);
//   cv::waitKey(0);

//   // cv::imshow("yellow", yellow_map);
//   // cv::waitKey(0);
//   // vector<pair<double, int>> scores;
//   //
//   // // sort & print top-3
//   // for (int i = 0; i < 10; i++)
//   //   scores.emplace_back(rescale<tanh_layer>(res[i]), i);
//   //
//   // sort(scores.begin(), scores.end(), greater<pair<double, int>>());
//   //
//   // for (int i = 0; i < 3; i++)
//   //   cout << scores[i].second << "," << scores[i].first << endl;
// }

void detectImg(const std::string &imgPath, double threshold) {
  int patchSize = 25;
  int patchRadius = int((patchSize-1)/2);
  int inputWidth = 320;
  int heightUp = 90; //80;
  int heightDown = 180; //140;
  int inputHeight = heightDown-heightUp;

  int outputWidth  = inputWidth - (patchSize - 1);
  int outputHeight  = inputHeight - (patchSize - 1);

  cv::Rect roi;
  roi.x = 0;
  roi.y = heightUp;
  roi.width = inputWidth;
  roi.height = inputHeight;
  auto imgSource = cv::imread(imgPath);

  cv::Mat Q, disp, rectified, XYZ, img;
  reconstruction(imgSource, Q, disp, rectified, XYZ);

  int index;
  std::string filename, savePath;

  index = imgPath.find_last_of('/');
  filename = imgPath.substr(index+1);
  savePath = imgPath.substr(0,index-7)+"/rectified/"+filename;
  cv::imwrite(savePath, rectified);

  cv::resize(rectified, img, cv::Size(320, 180));
  auto patchImg = img(roi);

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convertImage(patchImg, inputWidth, inputHeight, data);

  // recognize
  auto prob = nn.predict(data);

  cv::Mat probMap = cv::Mat::zeros(outputHeight, outputWidth, CV_64FC4);
  for (int c = 0; c < 4; ++c)
    for (int y = 0; y < outputHeight; ++y)
      for (int x = 0; x < outputWidth; ++x)
        probMap.at<cv::Vec4d>(y, x)[c] = prob[c * outputWidth * outputHeight + y * outputWidth + x];
         
  cv::Vec4d probSoftmax(4);
  cv::Mat probMapSoftmax = cv::Mat::zeros(outputHeight, outputWidth, CV_64F);
  cv::Mat probMapIndex = cv::Mat::zeros(outputHeight, outputWidth, CV_32S);
  cv::Mat probMapSplit[3] = cv::Mat::zeros(outputHeight, outputWidth, CV_64F);
  for (int y = 0; y < outputHeight; ++y){
    for (int x = 0; x < outputWidth; ++x){
      softmax(probMap.at<cv::Vec4d>(y, x), probSoftmax);
      for (int c = 0; c < 3; ++c)
        if(probSoftmax[c+1] > threshold){
          probMapSoftmax.at<double>(y, x) = probSoftmax[c+1];
          probMapIndex.at<int>(y, x) = c+1;
          probMapSplit[c].at<double>(y, x) = probSoftmax[c+1];
        }
    }
  }

  std::vector <cv::Point> cone;
  cv::Point position, positionShift = cv::Point(patchRadius, patchRadius+heightUp);
  int label;
  cone = imRegionalMax(probMapSoftmax, 10, threshold, 20);

  // std::ofstream savefile;
  // int index2 = filename.find_last_of('.');
  // savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".csv";
  // std::cout << savePath << std::endl;
  // savefile.open(savePath);
  cv::Vec3f point3D;
  if (cone.size()>0){
    for(size_t i=0; i<cone.size(); i++){
      position = (cone[i] + positionShift)*2;
      label = probMapIndex.at<int>(cone[i]);
      if (label == 1){
        cv::circle(rectified, position, 1, {255, 0, 0}, -1);
      }
      if (label == 2){
        cv::circle(rectified, position, 1, {0, 255, 255}, -1);
      }
      if (label == 3){
        cv::circle(rectified, position, 1, {0, 0, 255}, -1);
      }
      point3D = XYZ.at<cv::Vec3f>(position);
      std::cout << position << " " << label << " " << point3D << std::endl;
      // savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+std::to_string(label)+","+std::to_string(point3D[0])+","+std::to_string(point3D[1])+","+std::to_string(point3D[2])+"\n"; 
    }
  }
  // savefile.close();
  
  // cv::namedWindow("probMapSoftmax", cv::WINDOW_NORMAL);
  // cv::imshow("probMapSoftmax", probMapSoftmax);
  // cv::namedWindow("img", cv::WINDOW_NORMAL);
  // cv::imshow("img", rectified);
  // cv::waitKey(0);

  cv::namedWindow("rectified", cv::WINDOW_NORMAL);
  cv::imshow("rectified", rectified);
  cv::namedWindow("0", cv::WINDOW_NORMAL);
  cv::imshow("0", probMapSplit[0]);
  cv::namedWindow("1", cv::WINDOW_NORMAL);
  cv::imshow("1", probMapSplit[1]);
  cv::namedWindow("2", cv::WINDOW_NORMAL);
  cv::imshow("2", probMapSplit[2]);
  cv::waitKey(0);

  savePath = imgPath.substr(0,index-7)+"/results/"+filename;
  cv::imwrite(savePath, rectified);
  // std::cout << savePath << std::endl;
}

void detectAllImg(const std::string &modelPath, const std::string &imgFolderPath, double threshold){
  constructNetwork(modelPath, 320, 90);
  boost::filesystem::path dpath(imgFolderPath);
  BOOST_FOREACH(const boost::filesystem::path& imgPath, std::make_pair(boost::filesystem::directory_iterator(dpath), boost::filesystem::directory_iterator())) {
    std::cout << imgPath.string() << std::endl;
    
    auto startTime = std::chrono::system_clock::now();
	detectImg(imgPath.string(), threshold);
	auto endTime = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = endTime-startTime;
	std::cout << "Time: " << diff.count() << " s\n";
  }
}

int main(int argc, char **argv) {
  detectAllImg(argv[1], argv[2], atof(argv[3]));
}
