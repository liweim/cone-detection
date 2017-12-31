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
using namespace boost::filesystem;
using namespace tiny_dnn;
using namespace std;

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

// void mat2array(cv::Mat _mat, tiny_dnn::vec_t &ovect){
//     switch(_mat.channels()) {
//         case 1: {
//             tiny_dnn::float_t *ptr = _mat.ptr<tiny_dnn::float_t>(0);
//             ovect = tiny_dnn::vec_t(ptr, ptr + _mat.cols * _mat.rows );
//         } break;
//         case 3: {
//             std::vector<cv::Mat> _vmats;
//             cv::split(_mat, _vmats);
//             for(int i = 0; i < 3; i++) {
//                 cv::Mat _chanmat = _vmats[i];
//                 tiny_dnn::float_t *ptr = _chanmat.ptr<tiny_dnn::float_t>(0);
//                 ovect.insert(ovect.end(), ptr, ptr + _mat.cols * _mat.rows);
//             }
//         } break;
//     }
//     // for(int i=0;i<ovect.size();i++){
//     //   std::cout<< ovect[i]<<std::endl;
//     // }
// }

// convert all images found in directory to vec_t
void load_data(const string& directory,
                    double minv,
                    double maxv,
                    int w,
                    int h,
                    vector<vec_t>& train_imgs,
                    vector<label_t>& train_labels,
                    vector<vec_t>& test_imgs,
                    vector<label_t>& test_labels)
{
    path dpath(directory);
    int label_id;
    vec_t data;
    double random;

    BOOST_FOREACH(const path& label_path, make_pair(directory_iterator(dpath), directory_iterator())) {
        //if (is_directory(p)) continue;
        BOOST_FOREACH(const path& img_path, make_pair(directory_iterator(label_path), directory_iterator())) {
          label_id = stoi(label_path.filename().string());
          convert_image(img_path.string(), minv, maxv, w, h, data);
          // cv::Mat img = cv::imread(img_path.string());
          // img.convertTo(img, CV_32FC3);
          // img /= 255.0;
          // cv::resize(img, img, cv::Size(w, h));
          // mat2array(img, data);

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

template <typename N>
void construct_net(N &nn, core::backend_t backend_type) {
  using conv    = convolutional_layer;
  using dropout = dropout_layer;
  using pool    = max_pooling_layer;
  using fc      = fully_connected_layer;
  using relu    = relu_layer;
  using softmax = softmax_layer;
  using sigmoid = sigmoid_layer;

  // const size_t n_fmaps  = 32;  // number of feature maps for upper layer
  // const size_t n_fmaps2 = 64;  // number of feature maps for lower layer
  // const size_t n_fc     = 64;  // number of hidden units in fc layer
  //
  // nn << conv(32, 32, 5, 3, n_fmaps, padding::same, true, 1, 1,
  //            backend_type)                      // C1
  //    << pool(32, 32, n_fmaps, 2, backend_type)  // P2
  //    << relu()                                  // activation
  //    << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same, true, 1, 1,
  //            backend_type)                      // C3
  //    << pool(16, 16, n_fmaps, 2, backend_type)  // P4
  //    << relu()                                  // activation
  //    << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same, true, 1, 1,
  //            backend_type)                                // C5
  //    << pool(8, 8, n_fmaps2, 2, backend_type)             // P6
  //    << relu()                                            // activation
  //    << fc(4 * 4 * n_fmaps2, n_fc, true, backend_type)    // FC7
  //    << relu()                                            // activation
  //    << fc(n_fc, 3, true, backend_type) << softmax(3);  // FC10

  // nn << conv(32, 32, 5, 3, 32, padding::same)  // C1
  //    << pool(32, 32, 32, 2)                              // P2
  //    << relu(16, 16, 32)                                 // activation
  //    << conv(16, 16, 5, 32, 32, padding::same)  // C3
  //    << pool(16, 16, 32, 2)                                    // P4
  //    << relu(8, 8, 32)                                        // activation
  //    << conv(8, 8, 5, 32, 64, padding::same)  // C5
  //    << pool(8, 8, 64, 2)                                    // P6
  //    << relu(4, 4, 64)                                       // activation
  //    << conv(4, 4, 5, 64, 3, padding::same)
  //    << pool(4, 4, 3, 4)
  //    << softmax(3);

  const size_t input_size  = 25;
  nn << conv(input_size, input_size, 7, 3, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-6, input_size-6, 7, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-12)*(input_size-12)*32, 0.25)
     << conv(input_size-12, input_size-12, 5, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-16, input_size-16, 5, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     //<< dropout((input_size-20)*(input_size-20)*32, 0.25)
     << conv(input_size-20, input_size-20, 3, 32, 32, padding::valid, true, 1, 1, backend_type) << relu()
     << conv(input_size-22, input_size-22, 3, 32, 3, padding::valid, true, 1, 1, backend_type) << softmax(3);

   for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer:" << i << "\n";
        cout << "layer type:" << nn[i]->layer_type() << "\n";
        cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }
}

void train_cifar10(string data_dir_path,
                   double learning_rate,
                   const int n_train_epochs,
                   const int n_minibatch,
                   core::backend_t backend_type,
                   ostream &log) {
  // specify loss-function and learning strategy
  network<sequential> nn;
  adam optimizer;

  construct_net(nn, backend_type);

  cout << "load models..." << endl;

  // load cifar dataset
  vector<label_t> train_labels, test_labels;
  vector<vec_t> train_images, test_images;

  load_data(data_dir_path, 0, 1, 25, 25, train_images, train_labels, test_images, test_labels);

  cout << "start learning" << endl;

  progress_display disp(train_images.size());
  timer t;

  optimizer.alpha *=
    static_cast<float_t>(sqrt(n_minibatch) * learning_rate);

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << endl;
    ++epoch;
    result res = nn.test(test_images, test_labels);
    log << res.num_success << "/" << res.num_total << endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<cross_entropy>(optimizer, train_images, train_labels,
                                    n_minibatch, n_train_epochs,
                                    on_enumerate_minibatch, on_enumerate_epoch);

  cout << "end training." << endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(cout);
  // save networks
  ofstream ofs("models/sliding_window");
  ofs << nn;
}

static core::backend_t parse_backend_name(const string &name) {
  const array<const string, 5> names = {
    "internal", "nnpack", "libdnn", "avx", "opencl",
  };
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<core::backend_t>(i);
    }
  }
  return core::default_engine();
}

static void usage(const char *argv0) {
  cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 0.001"
            << " --epochs 100"
            << " --minibatch_size 32"
            << " --backend_type internal" << endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 0.01;
  int epochs                             = 5;
  string data_path                       = "";
  int minibatch_size                     = 32;
  core::backend_t backend_type = core::default_engine();

  if (argc == 2) {
    string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = string(argv[count + 1]);
    } else {
      cerr << "Invalid parameter specified - \"" << argname << "\""
                << endl;
      usage(argv[0]);
      return -1;
    }
  }
  if (data_path == "") {
    cerr << "Data path not specified." << endl;
    usage(argv[0]);
    return -1;
  }
  if (learning_rate <= 0) {
    cerr
      << "Invalid learning rate. The learning rate must be greater than 0."
      << endl;
    return -1;
  }
  if (epochs <= 0) {
    cerr << "Invalid number of epochs. The number of epochs must be "
                 "greater than 0."
              << endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 50000) {
    cerr
      << "Invalid minibatch size. The minibatch size must be greater than 0"
         " and less than dataset size (50000)."
      << endl;
    return -1;
  }
  cout << "Running with the following parameters:" << endl
            << "Data path: " << data_path << endl
            << "Learning rate: " << learning_rate << endl
            << "Minibatch size: " << minibatch_size << endl
            << "Number of epochs: " << epochs << endl
            << "Backend type: " << backend_type << endl
            << endl;
  try {
    train_cifar10(data_path, learning_rate, epochs, minibatch_size,
                  backend_type, cout);
  } catch (nn_error &err) {
    cerr << "Exception: " << err.what() << endl;
  }
}
