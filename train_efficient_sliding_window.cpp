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

int input_size = 25;

// void convert_image(cv::Mat img,
//                    int w,
//                    int h,
//                    tiny_dnn::vec_t& data){

//   cv::Mat resized, hsv[3];
//   cv::resize(img, resized, cv::Size(w, h));
//   cv::cvtColor(resized, resized, CV_RGB2HSV);
 
//   data.resize(w * h * 3);
//   for (size_t y = 0; y < h; ++y) {
//     for (size_t x = 0; x < w; ++x) {
//       data[y * w + x] = (resized.at<cv::Vec3b>(y, x)[0]) / 179.0;
//       data[w * h + y * w + x] = (resized.at<cv::Vec3b>(y, x)[1]) / 255.0;
//       data[2 * w * h + y * w + x] = (resized.at<cv::Vec3b>(y, x)[2]) / 255.0;
//     }
//   }
// }

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
  using softmax = tiny_dnn::softmax_layer;
  using dropout = tiny_dnn::dropout_layer;

  // nn << conv(input_size, input_size, 7, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-6, input_size-6, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((input_size-12)*(input_size-12)*16, 0.25)
  //    << conv(input_size-12, input_size-12, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-18, input_size-18, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((input_size-24)*(input_size-24)*16, 0.25)
  //    << conv(input_size-24, input_size-24, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-28, input_size-28, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((input_size-32)*(input_size-32)*16, 0.25)
  //    << conv(input_size-32, input_size-32, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-36, input_size-36, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << dropout((input_size-40)*(input_size-40)*16, 0.25)
  //    << conv(input_size-40, input_size-40, 3, 16, 128, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-42, input_size-42, 3, 128, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4);

  // nn << conv(input_size, input_size, 7, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-6, input_size-6, 7, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    // << dropout((input_size-12)*(input_size-12)*16, 0.25)
  //    << conv(input_size-12, input_size-12, 5, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-16, input_size-16, 5, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    // << dropout((input_size-20)*(input_size-20)*32, 0.25)
  //    << conv(input_size-20, input_size-20, 3, 32, 64, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
  //    << conv(input_size-22, input_size-22, 3, 64, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4);

  nn << conv(input_size, input_size, 5, 3, 8, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(input_size-4, input_size-4, 5, 8, 8, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((input_size-8)*(input_size-8)*8, 0.25)
     << conv(input_size-8, input_size-8, 5, 8, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(input_size-12, input_size-12, 5, 16, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((input_size-16)*(input_size-16)*16, 0.25)
     << conv(input_size-16, input_size-16, 3, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(input_size-18, input_size-18, 3, 32, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << dropout((input_size-20)*(input_size-20)*32, 0.25)
     << conv(input_size-20, input_size-20, 3, 32, 64, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh()
     << conv(input_size-22, input_size-22, 3, 64, 4, tiny_dnn::padding::valid, true, 1, 1, backend_type) << softmax(4);


   for (int i = 0; i < nn.depth(); i++) {
        std::cout << "#layer:" << i << "\n";
        std::cout << "layer type:" << nn[i]->layer_type() << "\n";
        std::cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        std::cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }
}

void train_network(std::string data_dir_path,
                   double learning_rate,
                   const int n_train_epochs,
                   const int n_minibatch,
                   tiny_dnn::core::backend_t backend_type,
                   std::ostream &log) {
  // specify loss-function and learning strategy
  tiny_dnn::network<tiny_dnn::sequential> nn;
  tiny_dnn::adam optimizer;

  construct_net(nn, backend_type);

  // std::ifstream ifs("efficient_sliding_window");
  // ifs >> nn;

  std::vector<tiny_dnn::vec_t> train_values, test_values, train_images, test_images;
  std::vector<tiny_dnn::label_t> train_labels, test_labels;

  load_data("tmp/"+data_dir_path, input_size, input_size, train_images, train_labels, train_values, test_images, test_labels, test_values);

  std::cout << "start learning" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *= static_cast<float_t>(sqrt(n_minibatch) * learning_rate);

  int epoch = 1;
  int loss_val_temp = 10000;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    // tiny_dnn::result train_res = nn.test(train_images, train_labels);
    // float_t loss_train = nn.get_loss<tiny_dnn::cross_entropy_multiclass>(train_images, train_values);
    // log << "Training accuracy: " << train_res.num_success << "/" << train_res.num_total << " = " << 100.0*train_res.num_success/train_res.num_total << "%, loss: " << loss_train << std::endl;

    tiny_dnn::result test_res = nn.test(test_images, test_labels);
    float_t loss_val = nn.get_loss<tiny_dnn::cross_entropy_multiclass>(test_images, test_values);
    log << "Validation accuracy: " <<test_res.num_success << "/" << test_res.num_total << " = " << 100.0*test_res.num_success/test_res.num_total << "%, loss: " << loss_val << std::endl;
    
    if(loss_val < 0){
      log << "Training crash!" << std::endl;
      return;
    }

    if(loss_val < loss_val_temp){
      loss_val_temp = loss_val;
      std::ofstream ofs ("models/"+data_dir_path);
      ofs << nn;
    }

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_values,
                                    n_minibatch, n_train_epochs,
                                    on_enumerate_minibatch, on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
}

static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {
    "internal", "nnpack", "libdnn", "avx", "opencl",
  };
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<tiny_dnn::core::backend_t>(i);
    }
  }
  return tiny_dnn::core::default_engine();
}

static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 0.001"
            << " --epochs 100"
            << " --minibatch_size 32"
            << " --backend_type internal" << std::endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 0.01;
  int epochs                             = 5;
  std::string data_path                       = "";
  int minibatch_size                     = 32;
  tiny_dnn::core::backend_t backend_type = parse_backend_name("internal");

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified." << std::endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 50000) {
    std::cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (50000)."
      << std::endl;
    return -1;
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl
            << std::endl;
  try {
    train_network(data_path, learning_rate, epochs, minibatch_size,
                  backend_type, std::cout);
  } catch (tiny_dnn::nn_error &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
}
