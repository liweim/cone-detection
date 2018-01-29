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

int PATCH_SIZE = 32;
// convert image to vec_t
void convert_image(const std::string& imagefilename,
                   double scale,
                   int w,
                   int h,
                   tiny_dnn::vec_t& data)
{
    auto img = cv::imread(imagefilename);
    cv::Mat hsv, channel[3];
    cv::cvtColor(img, hsv, CV_RGB2HSV);
    cv::split(hsv, channel);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(channel[0], resized, cv::Size(w, h));

    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
                   [=](uint8_t c) { return c * scale; });
}

// convert all images found in directory to vec_t
void load_data(const std::string& directory,
                    double scale,
                    int w,
                    int h,
                    std::vector<tiny_dnn::vec_t>& train_imgs,
                    std::vector<tiny_dnn::label_t>& train_labels,
                    std::vector<tiny_dnn::vec_t>& test_imgs,
                    std::vector<tiny_dnn::label_t>& test_labels)
{
    boost::filesystem::path dpath(directory);
    int label_id;
    tiny_dnn::vec_t data;
    double random;

    BOOST_FOREACH(const boost::filesystem::path& label_path, std::make_pair(boost::filesystem::directory_iterator(dpath), boost::filesystem::directory_iterator())) {
        //if (is_directory(p)) continue;
        BOOST_FOREACH(const boost::filesystem::path& img_path, std::make_pair(boost::filesystem::directory_iterator(label_path), boost::filesystem::directory_iterator())) {
          label_id = stoi(label_path.filename().string());
          convert_image(img_path.string(), scale, w, h, data);

          random = (double)rand()/(double)RAND_MAX;
          if (random < 0.7){
            train_labels.push_back(label_id);
            train_imgs.push_back(data);
          }
          else{
            test_labels.push_back(label_id);
            test_imgs.push_back(data);
          }

      }
    }
}

// void convert_image(const string &imagefilename,
//                    double minv,
//                    double maxv,
//                    int w,
//                    int h,
//                    vec_t &data) {
//
//   image<> img(imagefilename, tiny_dnn::image_type::rgb);
//   img = resize_image(img, w, h);
//   data.resize(img.width() * img.height() * img.depth());
//   for (size_t c = 0; c < img.depth(); ++c) {
//     for (size_t y = 0; y < img.height(); ++y) {
//       for (size_t x = 0; x < img.width(); ++x) {
//         data[c * img.width() * img.height() + y * img.width() + x] =
//           (maxv - minv) * (img[y * img.width() + x + c]) / 255.0 + minv;
//       }
//     }
//   }
// }

// // convert all images found in directory to vec_t
// void load_data(const string& directory,
//                     double minv,
//                     double maxv,
//                     int w,
//                     int h,
//                     vector<vec_t>& train_imgs,
//                     vector<label_t>& train_labels,
//                     vector<vec_t>& test_imgs,
//                     vector<label_t>& test_labels)
// {
//     path dpath(directory);
//     int label_id;
//     vec_t data;
//     double random;
//
//     BOOST_FOREACH(const path& label_path, make_pair(directory_iterator(dpath), directory_iterator())) {
//         //if (is_directory(p)) continue;
//         BOOST_FOREACH(const path& img_path, make_pair(directory_iterator(label_path), directory_iterator())) {
//           label_id = stoi(label_path.filename().string());
//           convert_image(img_path.string(), minv, maxv, w, h, data);
//
//           random = (double)rand()/(double)RAND_MAX;
//           if (random < 0.7){
//             train_labels.push_back(label_id);
//             train_imgs.push_back(data);
//           }
//           else{
//             test_labels.push_back(label_id);
//             test_imgs.push_back(data);
//           }
//
//       }
//     }
// }

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

  nn << conv(PATCH_SIZE, PATCH_SIZE, 5, 1, n_fmaps, tiny_dnn::padding::same, true, 1, 1,
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
     << fc(n_fc, 4, true, backend_type) << softmax(4);  // FC10

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

  std::vector<tiny_dnn::label_t> train_labels, test_labels;
  std::vector<tiny_dnn::vec_t> train_images, test_images;

  load_data(data_dir_path, 1/255, PATCH_SIZE, PATCH_SIZE, train_images, train_labels, test_images, test_labels);

  std::cout << "start learning" << std::endl;

  tiny_dnn::progress_display disp(train_images.size());
  tiny_dnn::timer t;

  optimizer.alpha *=
    static_cast<float_t>(sqrt(n_minibatch) * learning_rate);

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    log << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels,
                                    n_minibatch, n_train_epochs,
                                    on_enumerate_minibatch, on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  // save networks
  std::ofstream ofs("models/sliding_window");
  ofs << nn;
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
  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

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
