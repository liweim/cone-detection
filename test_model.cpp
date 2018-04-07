/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <cstdlib>
#include <iostream>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

int patch_size = 25;

// void convert_image(cv::Mat img,
//                    int w,
//                    int h,
//                    tiny_dnn::vec_t& data){

//   cv::Mat resized;
//   cv::resize(img, resized, cv::Size(w, h));
//   data.resize(w * h * 3);
//   for (size_t c = 0; c < 3; ++c) {
//     for (size_t y = 0; y < h; ++y) {
//       for (size_t x = 0; x < w; ++x) {
//         data[c * w * h + y * w + x] =
//           resized.at<cv::Vec3b>(y, x)[c] / 255.0;
//       }
//     }
//   }
// }

void convert_image(cv::Mat img,
                   int w,
                   int h,
                   tiny_dnn::vec_t& data){

  cv::Mat resized, hsv[3];
  cv::resize(img, resized, cv::Size(w, h));
  cv::cvtColor(resized, resized, CV_RGB2HSV);
 
  data.resize(w * h * 3);
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      data[y * w + x] = (resized.at<cv::Vec3b>(y, x)[0]-75) / 179.0;
      data[w * h + y * w + x] = (resized.at<cv::Vec3b>(y, x)[1]-46) / 255.0;
      data[2 * w * h + y * w + x] = (resized.at<cv::Vec3b>(y, x)[2]-107) / 255.0;
    }
  }
}

// convert all images found in directory to vec_t
void load_data(const std::string& directory,
                    int w,
                    int h,
                    std::vector<tiny_dnn::vec_t>& train_imgs,
                    std::vector<tiny_dnn::label_t>& train_labels,
                    std::vector<tiny_dnn::vec_t>& train_values,
                    std::vector<tiny_dnn::vec_t>& test_imgs,
                    std::vector<tiny_dnn::label_t>& test_labels,
                    std::vector<tiny_dnn::vec_t>& test_values)
{
    boost::filesystem::path trainPath(directory+"/train");
    boost::filesystem::path testPath(directory+"/test");
    int label;

    tiny_dnn::vec_t data, value;

    BOOST_FOREACH(const boost::filesystem::path& labelPath, std::make_pair(boost::filesystem::directory_iterator(trainPath), boost::filesystem::directory_iterator())) {
        //if (is_directory(p)) continue;
        BOOST_FOREACH(const boost::filesystem::path& imgPath, std::make_pair(boost::filesystem::directory_iterator(labelPath), boost::filesystem::directory_iterator())) {
          label = stoi(labelPath.filename().string());
          value = {0,0,0,0};
          value[label] = 1;
          auto img = cv::imread(imgPath.string());

          convert_image(img, w, h, data);
          train_values.push_back(value);
          train_labels.push_back(label);
          train_imgs.push_back(data);
      }
    }
    BOOST_FOREACH(const boost::filesystem::path& labelPath, std::make_pair(boost::filesystem::directory_iterator(testPath), boost::filesystem::directory_iterator())) {
        //if (is_directory(p)) continue;
        BOOST_FOREACH(const boost::filesystem::path& imgPath, std::make_pair(boost::filesystem::directory_iterator(labelPath), boost::filesystem::directory_iterator())) {
          label = stoi(labelPath.filename().string());
          value = {0,0,0,0};
          value[label] = 1;
          auto img = cv::imread(imgPath.string());

          convert_image(img, w, h, data);
          test_values.push_back(value);
          test_labels.push_back(label);
          test_imgs.push_back(data);
      }
    }
    std::cout << "loaded data" << std::endl;
}

template <typename N>
void construct_net(N &nn, tiny_dnn::core::backend_t backend_type) {
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using leaky_relu    = tiny_dnn::leaky_relu_layer;
  using softmax = tiny_dnn::softmax_layer;
  using dropout = tiny_dnn::dropout_layer;

  // nn << conv(patch_size, patch_size, 7, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-6, patch_size-6, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((patch_size-12)*(patch_size-12)*16, 0.25)
  //    << conv(patch_size-12, patch_size-12, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-18, patch_size-18, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((patch_size-24)*(patch_size-24)*16, 0.25)
  //    << conv(patch_size-24, patch_size-24, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-28, patch_size-28, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((patch_size-32)*(patch_size-32)*16, 0.25)
  //    << conv(patch_size-32, patch_size-32, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-36, patch_size-36, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((patch_size-40)*(patch_size-40)*16, 0.25)
  //    << conv(patch_size-40, patch_size-40, 3, 16, 128, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-42, patch_size-42, 3, 128, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4);

  // nn << conv(patch_size, patch_size, 7, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-6, patch_size-6, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    // << dropout((patch_size-12)*(patch_size-12)*16, 0.25)
  //    << conv(patch_size-12, patch_size-12, 5, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-16, patch_size-16, 5, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    // << dropout((patch_size-20)*(patch_size-20)*32, 0.25)
  //    << conv(patch_size-20, patch_size-20, 3, 32, 64, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(patch_size-22, patch_size-22, 3, 64, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4);

  nn << conv(patch_size, patch_size, 5, 3, 8, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(patch_size-4, patch_size-4, 5, 8, 8, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((patch_size-8)*(patch_size-8)*8, 0.25)
     << conv(patch_size-8, patch_size-8, 5, 8, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(patch_size-12, patch_size-12, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((patch_size-16)*(patch_size-16)*16, 0.25)
     << conv(patch_size-16, patch_size-16, 3, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(patch_size-18, patch_size-18, 3, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((patch_size-20)*(patch_size-20)*32, 0.25)
     << conv(patch_size-20, patch_size-20, 3, 32, 64, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(patch_size-22, patch_size-22, 3, 64, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4); 
}

void train_network(std::string data_path, std::string model_path) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;

  construct_net(nn, tiny_dnn::core::backend_t::internal);

  std::ifstream ifs("models/"+model_path);
  ifs >> nn;

  for (int i = 0; i < nn.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << nn[i]->layer_type() << "\n";
        std::cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        std::cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

  std::vector<tiny_dnn::vec_t> train_values, test_values, train_images, test_images;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;

  load_data("tmp/"+data_path, patch_size, patch_size, train_images, train_labels, train_values, test_images, test_labels, test_values);

  tiny_dnn::result train_res = nn.test(train_images, train_labels);
  float_t loss_train = nn.get_loss<tiny_dnn::cross_entropy_multiclass>(train_images, train_values);
  std::cout << "Training accuracy: " << train_res.num_success << "/" << train_res.num_total << " = " << 100.0*train_res.num_success/train_res.num_total << "%, loss: " << loss_train << std::endl;

  tiny_dnn::result test_res = nn.test(test_images, test_labels);
  float_t loss_val = nn.get_loss<tiny_dnn::cross_entropy_multiclass>(test_images, test_values);
  std::cout << "Validation accuracy: " <<test_res.num_success << "/" << test_res.num_total << " = " << 100.0*test_res.num_success/test_res.num_total << "%, loss: " << loss_val << std::endl;
}

int main(int argc, char **argv) {
  train_network(argv[1], argv[2]);
}
