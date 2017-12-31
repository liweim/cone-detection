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
void construct_net(N &nn, tiny_dnn::core::backend_t backend_type) {
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using relu    = tiny_dnn::relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  const size_t n_fmaps  = 32;  // number of feature maps for upper layer
  const size_t n_fmaps2 = 64;  // number of feature maps for lower layer
  const size_t n_fc     = 64;  // number of hidden units in fc layer

  nn << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same, true, 1, 1,
             backend_type)                      // C1
     << pool(32, 32, n_fmaps, 2, backend_type)  // P2
     << relu()                                  // activation
     << conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same, true, 1, 1,
             backend_type)                      // C3
     << pool(16, 16, n_fmaps, 2, backend_type)  // P4
     << relu()                                  // activation
     << conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same, true, 1, 1,
             backend_type)                                // C5
     << pool(8, 8, n_fmaps2, 2, backend_type)             // P6
     << relu()                                            // activation
     << fc(4 * 4 * n_fmaps2, n_fc, true, backend_type)    // FC7
     << relu()                                            // activation
     << fc(n_fc, 3, true, backend_type) << softmax(3);  // FC10

  //  for (int i = 0; i < nn.depth(); i++) {
  //       cout << "#layer:" << i << "\n";
  //       cout << "layer type:" << nn[i]->layer_type() << "\n";
  //       cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
  //       cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
  //   }
}

void recognize(const std::string &dictionary, const std::string &src_filename) {
  tiny_dnn::network<tiny_dnn::sequential> nn;

  construct_net(nn, tiny_dnn::core::default_engine());

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> nn;

  cv::Mat img = cv::imread(src_filename);
  cv::resize(img, img, cv::Size(32, 32));

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(src_filename, 0, 1.0, 32, 32, data);

  // recognize
  auto prob = nn.predict(data);
  std::cout << "yellow: " << prob[1] << ", blue: " << prob[2] << std::endl;

  // std::vector<pair<double, int>> scores;
  // // sort & print top-3
  // for (int i = 0; i < 10; i++)
  //   scores.emplace_back(rescale<tanh_layer>(res[i]), i);
  //
  // std::sort(scores.begin(), scores.end(), std::greater<pair<double, int>>());
  //
  // for (int i = 0; i < 3; i++)
  //   std::cout << scores[i].second << "," << scores[i].first << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "please specify image file";
    return 0;
  }
  recognize("models/sliding_window", argv[1]);
}
