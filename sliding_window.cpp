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

using namespace tiny_dnn;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   vec_t &data) {

  image<> img(imagefilename, tiny_dnn::image_type::rgb);
  image<> resized = resize_image(img, w, h);
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
void construct_net(N &nn, core::backend_t backend_type, const size_t input_size) {
  using conv    = convolutional_layer;
  using dropout = dropout_layer;
  using pool    = max_pooling_layer;
  using fc      = fully_connected_layer;
  using relu    = relu_layer;
  using softmax = softmax_layer;
  using sigmoid = sigmoid_layer;

  // const size_t n_fmaps  = 32;  ///< number of feature maps for upper layer
  // const size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
  // const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer

  // nn << conv(32, 32, 5, 3, n_fmaps, padding::same)  // C1
  //    << pool(32, 32, n_fmaps, 2)                              // P2
  //    << relu(16, 16, n_fmaps)                                 // activation
  //    << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)  // C3
  //    << pool(16, 16, n_fmaps, 2)                                    // P4
  //    << relu(8, 8, n_fmaps)                                        // activation
  //    << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)  // C5
  //    << pool(8, 8, n_fmaps2, 2)                                    // P6
  //    << relu(4, 4, n_fmaps2)                                       // activation
  //    << fc(4 * 4 * n_fmaps2, n_fc)                                 // FC7
  //    << fc(n_fc, 3) << softmax(3);                               // FC10

  nn << conv(input_size, input_size, 7, 3, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-6, input_size-6, 7, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-12)*(input_size-12)*32, 0.25)
     << conv(input_size-12, input_size-12, 5, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-16, input_size-16, 5, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-20)*(input_size-20)*32, 0.25)
     << conv(input_size-20, input_size-20, 3, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-22, input_size-22, 3, 32, 3, padding::valid, true, 1, 1, backend_type);

  // const size_t input_size  = 25;
  // nn << conv(input_size, input_size, 7, 3, 32, padding::valid, true, 1, 1, backend_type) << relu()
  //    << conv(input_size-6, input_size-6, 7, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
  //    << dropout((input_size-12)*(input_size-12)*32, 0.25)
  //    << conv(input_size-12, input_size-12, 5, 32, 64, padding::valid, true, 1, 1, backend_type) << relu()
  //    << conv(input_size-16, input_size-16, 5, 64, 64, padding::valid, true, 1, 1, backend_type) << relu()
  //    << dropout((input_size-20)*(input_size-20)*64, 0.25)
  //    << conv(input_size-20, input_size-20, 3, 64, 128, padding::valid, true, 1, 1, backend_type) << relu()
  //    << conv(input_size-22, input_size-22, 3, 128, 3, padding::valid, true, 1, 1, backend_type) << softmax(3);

  //  for (int i = 0; i < nn.depth(); i++) {
  //       cout << "#layer:" << i << "\n";
  //       cout << "layer type:" << nn[i]->layer_type() << "\n";
  //       cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
  //       cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
  //   }
}

void softmax(vec_t &x, vec_t &y){
    float_t alpha = *max_element(x.begin(), x.end());
    float_t denominator(0);
    for (size_t j = 0; j < x.size(); j++) {
      y[j] = exp(x[j] - alpha);
      denominator += y[j];
    }
    for (size_t j = 0; j < x.size(); j++) {
      y[j] /= denominator;
    }
  }

vector <cv::Point> GetLocalMaxima(const cv::Mat Src,int MatchingSize, int Threshold){
  vector <cv::Point> vMaxLoc(0);
  vMaxLoc.reserve(100); // Reserve place for fast access
  cv::Mat ProcessImg = Src.clone();
  int W = Src.cols;
  int H = Src.rows;
  int SearchWidth  = W - MatchingSize;
  int SearchHeight = H - MatchingSize;
  int MatchingSquareCenter = MatchingSize/2;

  cv::GaussianBlur(ProcessImg, ProcessImg, cv::Size(3, 3), 0.5, 0.5);
  cout << ProcessImg << endl;
  uchar* pProcess = (uchar *) ProcessImg.data; // The pointer to image Data

  int Shift = MatchingSquareCenter * ( W + 1);
  int k = 0;

  for(int y=0; y < SearchHeight; ++y)
  {
    int m = k + Shift;
    for(int x=0;x < SearchWidth ; ++x)
    {
      if (pProcess[m++] >= Threshold)
      {
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
  cout<<vMaxLoc<<endl;
  return vMaxLoc;
}

void recognize(const string &dictionary, const string &src_filename) {
  size_t patch_size  = 26;
  int radius = (patch_size-1)/2;
  int pad_size  = patch_size - 25 + 1;

  network<sequential> nn;

  construct_net(nn, core::default_engine(), patch_size);

  // load nets
  ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  vec_t data;
  convert_image(src_filename, 0, 1.0, patch_size, patch_size, data);

  // recognize
  auto prob = nn.predict(data);


  cv::Mat prob_map[2] = cv::Mat::zeros(patch_size, patch_size, CV_32FC1);
  int r, c;
  for(int i=0; i<prob.size(); i+=3){
    vec_t prob_temp;
    prob_temp.push_back(prob[i]);
    prob_temp.push_back(prob[i+1]);
    prob_temp.push_back(prob[i+2]);
    softmax(prob_temp, prob_temp);
    cout << "yellow: " << prob_temp[1] << ", blue: " << prob_temp[2] << endl;

    r = i/3 / pad_size;
    c = i/3 % pad_size;
    prob_map[0].at<float_t>(radius+r, radius+c) = float_t(prob_temp[1]);
    prob_map[1].at<float_t>(radius+r, radius+c) = float_t(prob_temp[2]);
  }

  cv::Mat yellow_map, blue_map;
  vector <cv::Point> yellow, blue;

  yellow =  GetLocalMaxima(prob_map[0], 3, 0.5);
  blue =  GetLocalMaxima(prob_map[1], 3, 0.5);

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
    cout << "please specify image file";
    return 0;
  }
  recognize("models/sliding_window", argv[1]);
}
