#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <tiny_dnn/tiny_dnn.h>

tiny_dnn::network<tiny_dnn::sequential> m_slidingWindow;

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

void slidingWindow(const std::string &dictionary) {
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

void siftCNN(cv::Mat rectified, std::vector<cv::KeyPoint> pts, std::vector<int>& outputs){
  //Given RoI by SIFT detector and detected by CNN
  float_t threshold = 0.7;
  int radius = 12;
  
  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Vec3i> porperty;
  outputs.clear();

  for(size_t i = 0; i < pts.size(); i++){
    int x = pts[i].pt.x;
    int y = pts[i].pt.y;

	cv::Rect roi;
	roi.x = std::max(x - radius, 0);
	roi.y = std::max(y - radius, 0);
	roi.width = std::min(x + radius, rectified.cols) - roi.x;
	roi.height = std::min(y + radius, rectified.rows) - roi.y;

	//cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
	// // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
	//cv::namedWindow("roi", cv::WINDOW_NORMAL);
	//cv::imshow("roi", img);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
	if (0 > roi.x || 0 > roi.width || roi.x + roi.width > rectified.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > rectified.rows){
		std::cout << "Wrong roi!" << std::endl;
		outputs.push_back(-1);
	}
	else{
		auto patchImg = rectified(roi);
		tiny_dnn::vec_t data;
		convertImage(patchImg, 25, 25, data);
		inputs.push_back({data});
		outputs.push_back(0);
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
      outputs[verifiedIndex[i]] = maxIndex;
      int x = int(porperty[i][0]);
      int y = int(porperty[i][1]);
      float_t radius = 2;

      std::string labels[] = {"blue", "yellow", "orange", "big orange"};
      if (maxIndex == 0 || maxProb < threshold){
        std::cout << "No cone detected" << std::endl;
        cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,0,0), -1);
      } 
      else{
        std::cout << "Find one " << labels[maxIndex-1] << " cone"<< std::endl;
        if (labels[maxIndex-1] == "blue")
          cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (255,0,0), -1);
        else if (labels[maxIndex-1] == "yellow")
          cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,255,255), -1);
        else if (labels[maxIndex-1] == "orange")
          cv::circle(rectified, cv::Point (x,y), radius, cv::Scalar (0,165,255), -1);
        else if (labels[maxIndex-1] == "big orange")
          cv::circle(rectified, cv::Point (x,y), radius*2, cv::Scalar (0,0,255), -1);
      }
    }
  }

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", rectified);
  // cv::waitKey(0);
  // cv::destroyAllWindows();

  // for(size_t i = 0; i < pts.size(); i++)
  //   std::cout << i << ": " << outputs[i] << std::endl;
}


int main( int argc, char** argv )
{

	slidingWindow("models/all_rgb_best");
	for(int i = 0; i < 316; i++){
		auto startTime = std::chrono::system_clock::now();

		cv::Mat img = cv::imread("annotations/skidpad1/rectified/"+std::to_string(i)+".png");
		img = img.rowRange(150, 270);
		img.rowRange(0,12) = 0;
		img.rowRange(108,120) = 0;
		cv::Mat img_hsv;
		cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

		cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create(20);
		std::vector<cv::KeyPoint> keypoints;
		detector->detect(img_hsv, keypoints);
		
		cv::Mat Match;
		cv::drawKeypoints(img, keypoints, Match);

		// cv::namedWindow("cv::Match", cv::WINDOW_NORMAL);
		// cv::imshow("cv::Match", Match);
		// cv::waitKey(0);

		std::vector<int> outputs;
		siftCNN(img, keypoints, outputs);

		auto endTime = std::chrono::system_clock::now();
	  	std::chrono::duration<double> diff = endTime-startTime;
	  	std::cout << "Time: " << diff.count() << " s\n";
	}
}
