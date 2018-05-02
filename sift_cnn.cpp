#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <tiny_dnn/tiny_dnn.h>

tiny_dnn::network<tiny_dnn::sequential> m_slidingWindow;

void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(); 
  sbm->setBlockSize(17);
  sbm->setNumDisparities(32);

  sbm->compute(grayL, grayR, disp);
  cv::normalize(disp, disp, 0, 255, 32, CV_8U);
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

  // cv::imwrite("tmp/imgL.png", imgL);
  // cv::imwrite("tmp/imgR.png", imgR);
  // return;

  blockMatching(disp, imgL, imgR);

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", disp);
  // cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t &data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

void slidingWindow(const std::string& dictionary) {
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using leaky_relu    = tiny_dnn::leaky_relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  m_slidingWindow << conv(25, 25, 4, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh() 
     // << dropout(22*22*16, 0.25)                    
     << pool(22, 22, 16, 2, backend_type)                               
     << conv(11, 11, 4, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh() 
     // << dropout(8*8*32, 0.25)                    
     << pool(8, 8, 32, 2, backend_type) 
     << fc(4 * 4 * 32, 128, true, backend_type) << leaky_relu()  
     << fc(128, 5, true, backend_type) << softmax(5);

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> m_slidingWindow;
}

void forwardDetectionRoI(const std::string& img_path){
  //Given RoI by SIFT detector and detected by CNN
  float_t threshold = 0.7;
  int radius = 12;
  
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Vec3i> porperty;
  // std::vector<int> outputs;

  cv::Mat imgSource = cv::imread(img_path);
	cv::Mat Q, disp, XYZ, img;
	reconstruction(imgSource, Q, disp, img, XYZ);
	// img = img.rowRange(176, 376);
	img = img.rowRange(140, 270);
	img.rowRange(0,24) = 0;
	img.rowRange(106,130) = 0;
	cv::Mat img_hsv;
	cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

	cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(15);
	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img_hsv, keypoints);

	cv::Mat Match;
	cv::drawKeypoints(img, keypoints, Match);

	// cv::namedWindow("cv::Match", cv::WINDOW_NORMAL);
	// cv::imshow("cv::Match", Match);
	// cv::waitKey(0);

	cv::resize(img, img, cv::Size(336, 65));
	// cv::resize(img, img, cv::Size(336, 100));

  for(size_t i = 0; i < keypoints.size(); i++){
    int x = keypoints[i].pt.x/2;
    int y = keypoints[i].pt.y/2;

	cv::Rect roi;
	roi.x = std::max(x - radius, 0);
	roi.y = std::max(y - radius, 0);
	roi.width = std::min(x + radius, img.cols) - roi.x;
	roi.height = std::min(y + radius, img.rows) - roi.y;

	//cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
	// // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
	//cv::namedWindow("roi", cv::WINDOW_NORMAL);
	//cv::imshow("roi", img);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
	if (0 > roi.x || 0 > roi.width || roi.x + roi.width > img.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > img.rows){
		std::cout << "Wrong roi!" << std::endl;
		// outputs.push_back(-1);
	}
	else{
		auto patchImg = img(roi);
		tiny_dnn::vec_t data;
		convertImage(patchImg, 25, 25, data);
		inputs.push_back({data});
		// outputs.push_back(0);
		verifiedIndex.push_back(i);
		porperty.push_back(cv::Vec3i(x,y,radius));
	}
  }
  
  if(inputs.size()>0){
    auto prob = m_slidingWindow.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      float_t maxProb = prob[i][0][0];
      for(size_t j = 1; j < 5; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }
      // outputs[verifiedIndex[i]] = maxIndex;
      int x = int(porperty[i][0]);
      int y = int(porperty[i][1]);
      float_t radius = 1;

      std::string labels[] = {"blue", "yellow", "orange", "big orange"};
      if (maxIndex == 0 || maxProb < threshold){
        std::cout << "No cone detected" << std::endl;
        cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0), -1);
      } 
      else{
        std::cout << "Find one " << labels[maxIndex-1] << " cone"<< std::endl;
        if (labels[maxIndex-1] == "blue")
          cv::circle(img, cv::Point (x,y), radius, cv::Scalar (255,0,0), -1);
        else if (labels[maxIndex-1] == "yellow")
          cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,255,255), -1);
        else if (labels[maxIndex-1] == "orange")
          cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,165,255), -1);
        else if (labels[maxIndex-1] == "big orange")
          cv::circle(img, cv::Point (x,y), radius*2, cv::Scalar (0,0,255), -1);
      }
    }
  }

  cv::namedWindow("disp", cv::WINDOW_NORMAL);
  cv::imshow("disp", img);
  cv::waitKey(0);
  // cv::destroyAllWindows();

  // for(size_t i = 0; i < pts.size(); i++)
  //   std::cout << i << ": " << outputs[i] << std::endl;
}

void extractFeature(){

}


int main( int argc, char** argv )
{

	slidingWindow("models/all_rgb");
	for(int i = 1; i < 316; i++){
		auto startTime = std::chrono::system_clock::now();
		forwardDetectionRoI("annotations/rainy/images/"+std::to_string(i)+".png");
		

		auto endTime = std::chrono::system_clock::now();
	  	std::chrono::duration<double> diff = endTime-startTime;
	  	std::cout << "Time: " << diff.count() << " s\n";
	}
}
