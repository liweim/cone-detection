/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

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
  image<> img(imagefilename, image_type::rgb);
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
void construct_net(N &nn, core::backend_t backend_type) {
  using conv    = convolutional_layer;
  using dropout = dropout_layer;
  using pool    = max_pooling_layer;
  using fc      = fully_connected_layer;
  using relu    = relu_layer;
  using softmax = softmax_layer;

  // const size_t n_fmaps  = 32;  ///< number of feature maps for upper layer
  // const size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
  // const size_t n_fc = 64;  ///< number of hidden units in fully-connected layer
  //
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
  //    << fc(n_fc, 10) << softmax(10);                               // FC10

  const size_t input_size  = 26;
  nn << conv(input_size, input_size, 7, 3, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-6, input_size-6, 7, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << dropout((input_size-12)*(input_size-12)*32, 0.25)
     << conv(input_size-12, input_size-12, 5, 32, 64, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-16, input_size-16, 5, 64, 64, padding::valid, true, 1, 1, backend_type) << relu()
     << dropout((input_size-20)*(input_size-20)*64, 0.25)
     << conv(input_size-20, input_size-20, 3, 64, 128, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-22, input_size-22, 3, 128, 3, padding::valid, true, 1, 1, backend_type);

  //  for (int i = 0; i < nn.depth(); i++) {
  //       cout << "#layer:" << i << "\n";
  //       cout << "layer type:" << nn[i]->layer_type() << "\n";
  //       cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
  //       cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
  //   }
}

void recognize(const string &dictionary, const string &src_filename) {
  network<sequential> nn;

  construct_net(nn, core::default_engine());

  // load nets
  ifstream ifs(dictionary.c_str());
  ifs >> nn;

  // convert imagefile to vec_t
  vec_t data;
  convert_image(src_filename, 0, 1.0, 26, 26, data);

  // recognize
  auto res = nn.predict(data);
  for(int i=0; i<res.size(); i++){
    cout << res[i] << endl;
  }

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
  recognize("cifar-weights", argv[1]);
}
