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

// convert image to vec_t
void convert_image(const string& imagefilename,
                   int w,
                   int h,
                   vector<vec_t>& data,
                   vector<label_t>& label)
{
  image<> img(imagefilename, image_type::rgb);
  img = resize_image(img, w, h);
  data.push_back(img.to_vec());
  label.push_back(0);
}

// convert all images found in directory to vec_t
void convert_images(const string& directory,
                    int w,
                    int h,
                    vector<vec_t>& data,
                    vector<label_t>& label)
{
    path dpath(directory);

    BOOST_FOREACH(const path& label_path, make_pair(directory_iterator(dpath), directory_iterator())) {
        //if (is_directory(p)) continue;
        BOOST_FOREACH(const path& img_path, make_pair(directory_iterator(label_path), directory_iterator())) {
          string label_id = label_path.filename().string();
          //cout << img_path.string() << " " << label_id << endl;
          image<> img(img_path.string(), image_type::rgb);
          img = resize_image(img, w, h);
          data.push_back(img.to_vec());
          label.push_back(stoi(label_id));
      }
    }
}

template <typename N>
void construct_net(N &nn, core::backend_t backend_type) {
  using conv    = convolutional_layer;
  using pool    = max_pooling_layer;
  using fc      = fully_connected_layer;
  using relu    = relu_layer;
  using softmax = softmax_layer;

  const size_t n_fmaps  = 32;  // number of feature maps for upper layer
  const size_t n_fmaps2 = 64;  // number of feature maps for lower layer
  const size_t n_fc     = 64;  // number of hidden units in fc layer

  nn << conv(32, 32, 5, 3, n_fmaps, padding::same, true, 1, 1,
             backend_type)                      // C1
     << pool(32, 32, n_fmaps, 2, backend_type)  // P2
     << relu()                                  // activation
     << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same, true, 1, 1,
             backend_type)                      // C3
     << pool(16, 16, n_fmaps, 2, backend_type)  // P4
     << relu()                                  // activation
     << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same, true, 1, 1,
             backend_type)                                // C5
     << pool(8, 8, n_fmaps2, 2, backend_type)             // P6
     << relu()                                            // activation
     << fc(4 * 4 * n_fmaps2, n_fc, true, backend_type)    // FC7
     << relu()                                            // activation
     << fc(n_fc, 10, true, backend_type) << softmax(10);  // FC10
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

  convert_images(data_dir_path, 32, 32, train_images, train_labels);
  convert_images(data_dir_path, 32, 32, test_images, test_labels);

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
  ofstream ofs("cifar-weights");
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
            << " --learning_rate 0.01"
            << " --epochs 30"
            << " --minibatch_size 10"
            << " --backend_type internal" << endl;
}

int main(int argc, char **argv) {
  double learning_rate                   = 0.01;
  int epochs                             = 30;
  string data_path                  = "";
  int minibatch_size                     = 10;
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
