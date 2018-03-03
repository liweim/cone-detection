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

void convert_image(cv::Mat img,
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
     << conv(width-22, height-22, 3, 32, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type);

  for (int i = 0; i < nn.depth(); i++) {
    std::cout << "#layer:" << i << "\n";
    std::cout << "layer type:" << nn[i]->layer_type() << "\n";
    std::cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
    std::cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
  }
}

void softmax(cv::Vec4f x, cv::Vec4f &y) {
  double min, max, denominator(0);
  cv::minMaxLoc(x, &min, &max);
  for (int j = 0; j < 4; j++) {
    y[j] = std::exp(x[j] - max);
    denominator += y[j];
  }
  for (int j = 0; j < 4; j++) {
    y[j] /= denominator;
  }
}


std::vector <cv::Point> imregionalmax(cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
    cv::Mat scratch = input.clone();
    std::cout<<scratch<<std::endl;
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
            locations.push_back(cv::Point(row, col));
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<float>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }

    std::cout<<locations<<std::endl;
    return locations;
}

void recognize(const std::string &dictionary, const std::string &img_path, double threshold) {
  int width = 100;
  int height = 100;
  int radius_width = (width-1)/2;
  int radius_height = (height-1)/2;
  int patch_size = 25;
  int patch_radius = (patch_size-1)/2;
  int output_width  = width - (patch_size - 1);
  int output_height  = height - (patch_size - 1);

  tiny_dnn::network<tiny_dnn::sequential> nn;

  construct_net(nn, tiny_dnn::core::default_engine(), width, height);

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  // tiny_dnn::image<> img(img_path, tiny_dnn::image_type::bgr);

  // manual roi
  // (453, 237, 0.96875, "orange")
  // (585, 211,	0.625,	"orange");
  // (343, 185,	0.25,	"yellow");
  // (521, 198,	0.375,	"yellow");
  // (634,	202,	0.375,	"yellow");
  // (625,	191,	0.34375,	"blue");
  // (396,	183,	0.34375,	"blue");

  int cx = 520;
  int cy = 210;
  double ratio = 2;
  std::string label = "orange";

  int length = ratio * patch_size;
  int radius = (length-1)/2;
  cv::Rect roi;
  roi.x = cx - radius;
  roi.y = cy - radius;
  roi.width = length;
  roi.height = length;
  auto img = cv::imread(img_path);
  auto patch_img = img(roi);
  cv::resize(patch_img, patch_img, cv::Size(width, height));

  cv::namedWindow("img", cv::WINDOW_NORMAL);
  cv::imshow("img", patch_img);
  cv::waitKey(0);
  cv::destroyAllWindows();

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(patch_img, width, height, data);

  // recognize
  auto prob = nn.predict(data);

  cv::Mat prob_map = cv::Mat::zeros(output_width, output_height, CV_32FC4);
  for (size_t c = 0; c < 4; ++c)
    for (size_t y = 0; y < output_height; ++y)
      for (size_t x = 0; x < output_width; ++x)
         prob_map.at<cv::Vec4f>(y, x)[c] = prob[c * output_width * output_height + y * output_width + x];

  cv::Vec4f prob_softmax(4);
  cv::Mat prob_map_softmax = cv::Mat::zeros(output_width, output_height, CV_32FC3);
  for (size_t y = 0; y < output_height; ++y)
    for (size_t x = 0; x < output_width; ++x){
      softmax(prob_map.at<cv::Vec4f>(y, x), prob_softmax);
      for (size_t c = 0; c < 3; ++c)
        if(prob_softmax[c+1] > threshold)
          prob_map_softmax.at<cv::Vec3f>(y, x)[c] = prob_softmax[c+1];
    }
  cv::Mat prob_map_split[3];
  cv::split(prob_map_softmax, prob_map_split);

  for (size_t c = 0; c < 3; ++c){
    cv::namedWindow("img", cv::WINDOW_NORMAL);
    cv::imshow("img", prob_map_split[c]);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }

  // std::vector <cv::Point> yellow, blue, orange;
  //
  // int nLocMax = 1;
  // int minDistBtwLocMax = 10;
  // yellow = imregionalmax(prob_map[0], nLocMax, threshold, minDistBtwLocMax);
  // blue = imregionalmax(prob_map[1], nLocMax, threshold, minDistBtwLocMax);
  // orange = imregionalmax(prob_map[2], nLocMax, threshold, minDistBtwLocMax);
  //
  // cv::Point cone;
  //
  // if (yellow.size()>0){
  //   for(int i=0; i<yellow.size(); i++){
  //     cone = yellow[i] + cv::Point(radius_width, radius_height);
  //     cv::circle(img, cone, 1, {0, 255, 255}, 1);
  //   }
  // }
  // if (blue.size()>0){
  //   for(int i=0; i<blue.size(); i++){
  //     cone = blue[i] + cv::Point(patch_radius, patch_radius);
  //     cv::circle(img, cone, 1, {255, 0, 0}, 1);
  //   }
  // }
  // if (orange.size()>0){
  //   for(int i=0; i<orange.size(); i++){
  //     cone = orange[i] + cv::Point(radius_width, radius_height);
  //     cv::circle(img, cone, 1, {0, 165, 255}, 1);
  //   }
  // }
  // cv::namedWindow("img", cv::WINDOW_NORMAL);
  // cv::imshow("img", img);
  // cv::waitKey(0);

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
  recognize("models/efficient_sliding_window", argv[1], atof(argv[2]));
}
