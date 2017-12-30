/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>
#include <typeinfo>

#include "tiny_dnn/tiny_dnn.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &src_filename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   tiny_dnn::vec_t &data) {

  tiny_dnn::image<> img(src_filename, tiny_dnn::image_type::rgb);
  tiny_dnn::image<> resized = tiny_dnn::resize_image(img, w, h);
  data.resize(resized.width() * resized.height() * resized.depth());
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t y = 0; y < resized.height(); ++y) {
      for (size_t x = 0; x < resized.width(); ++x) {
        data[c * resized.width() * resized.height() + y * resized.width() + x] =
          (maxv - minv) * (resized[y * resized.width() + x + c]) / 255.0 + minv;
      }
    }
  }
}

template <typename N>
void construct_net(N &nn, tiny_dnn::core::backend_t backend_type, int width, int height) {
  using conv    = tiny_dnn::convolutional_layer;
  using dropout = tiny_dnn::dropout_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;
  using sigmoid = tiny_dnn::sigmoid_layer;

  nn << conv(width, height, 7, 3, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-6, height-6, 7, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-12)*(input_size-12)*32, 0.25)
     << conv(width-12, height-12, 5, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-16, height-16, 5, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-20)*(input_size-20)*32, 0.25)
     << conv(width-20, height-20, 3, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << relu()
     << conv(width-22, height-22, 3, 32, 3, tiny_dnn::padding::valid, true, 1, 1, backend_type);

  //  for (int i = 0; i < nn.depth(); i++) {
  //       cout << "#layer:" << i << "\n";
  //       cout << "layer type:" << nn[i]->layer_type() << "\n";
  //       cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
  //       cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
  //   }
}

void softmax(std::vector<float_t> &x, std::vector<float_t> &y){
    float_t denominator(0);
    for (int j = 0; j < x.size(); j++) {
      y[j] = exp(x[j]);
      denominator += y[j];
    }
    for (int j = 0; j < x.size(); j++) {
      y[j] /= denominator;
    }
  }

std::vector <cv::Point> GetLocalMaxima(const cv::Mat Src,int MatchingSize, float_t Threshold){
  std::vector <cv::Point> vMaxLoc(0);
  vMaxLoc.reserve(100); // Reserve place for fast access
  cv::Mat ProcessImg = Src.clone();
  int W = Src.cols;
  int H = Src.rows;
  int SearchWidth  = W - MatchingSize;
  int SearchHeight = H - MatchingSize;
  int MatchingSquareCenter = MatchingSize/2;

  cv::GaussianBlur(ProcessImg, ProcessImg, cv::Size(3, 3), 0.5, 0.5);
  std::cout << ProcessImg << std::endl;
  float_t* pProcess = (float_t*) ProcessImg.data;

  int Shift = MatchingSquareCenter * ( W + 1);
  int k = 0;

  for(int y=0; y < SearchHeight; ++y)
  {
    int m = k + Shift;
    for(int x=0;x < SearchWidth ; ++x)
    {
      if (pProcess[m++] >= Threshold)
      {
        std::cout << pProcess[m] << std::endl;
        cv::Point LocMax;
        cv::Mat mROI(ProcessImg, cv::Rect(x,y,MatchingSize,MatchingSize));
        cv::minMaxLoc(mROI,NULL,NULL,NULL,&LocMax);
        if (LocMax.x == MatchingSquareCenter && LocMax.y == MatchingSquareCenter)
        {
          vMaxLoc.push_back(cv::Point( x+LocMax.x,y + LocMax.y ));
          // imshow("W1",mROI);cvWaitKey(0); //For gebug
        }
      }
    }
    k += W;
  }
  std::cout << vMaxLoc << std::endl;
  return vMaxLoc;
}

void recognize(const std::string &dictionary, const std::string &src_filename) {
  int width  = 25;
  int height = 25;
  int radius_width = (width-1)/2;
  int radius_height = (height-1)/2;
  int patch_size = 25;
  int pad_width  = width - patch_size + 1;
  int pad_height  = height - patch_size + 1;

  tiny_dnn::network<tiny_dnn::sequential> nn;

  construct_net(nn, tiny_dnn::core::default_engine(), width, height);

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;

  convert_image(src_filename, 0, 1.0, width, height, data);

  // recognize
  auto prob = nn.predict(data);

  cv::Mat prob_map[2] = cv::Mat::zeros(width, height, CV_32FC1);
  int r, c;
  for(int i=0; i<prob.size(); i+=3){
    std::vector<float_t> prob_temp(&prob[i], &prob[i+3]), prob_softmax(3);
    softmax(prob_temp, prob_softmax);
    std::cout << "yellow: " << prob_softmax[1] << ", blue: " << prob_softmax[2] << std::endl;

    r = i/3 / pad_width;
    c = i/3 % pad_height;
    prob_map[0].at<float_t>(radius_width+r, radius_height+c) = float_t(prob_softmax[1]);
    prob_map[1].at<float_t>(radius_width+r, radius_height+c) = float_t(prob_softmax[2]);
  }

  cv::Mat yellow_map, blue_map;
  std::vector <cv::Point> yellow, blue;

  yellow =  GetLocalMaxima(prob_map[0], 3, 0.6);
  blue =  GetLocalMaxima(prob_map[1], 3, 0.6);

  cv::Mat img = cv::imread(src_filename);
  cv::resize(img, img, cv::Size(width, height));
  if (yellow.size()>0){
    for(int i=0; i<yellow.size(); i++){
      cv::circle(img, yellow[i], 1, {0, 255, 255}, 1);
    }
  }
  if (blue.size()>0){
    for(int i=0; i<blue.size(); i++){
      cv::circle(img, blue[i], 1, {255, 0, 0}, 1);
    }
  }
  cv::namedWindow("img", cv::WINDOW_NORMAL);
  cv::imshow("img", img);
  cv::waitKey(0);
  // cv::imshow("yellow", yellow_map);
  // cv::waitKey(0);
  // vector<pair<double, int>> scores;
  //
  // // sort & print top-3
  // for (int i = 0; i < 10; i++)
  //   scores.emplace_back(rescale<tanh_layer>(res[i]), i);
  //
  // sort(scores.begin(), scores.end(), greater<pair<double, int>>());
  //
  // for (int i = 0; i < 3; i++)
  //   cout << scores[i].second << "," << scores[i].first << endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "please specify image file";
    return 0;
  }
  recognize("models/sliding_window", argv[1]);
}
