#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tiny_dnn/tiny_dnn.h>
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <dirent.h>
#include <boost/filesystem.hpp>

tiny_dnn::network<tiny_dnn::sequential> m_slidingWindow;

int PATCH_SIZE = 64;
// int WIDTH = 640;
// int HEIGHT = 360;
int WIDTH = 672;
int HEIGHT = 376;
bool ANNOTATE = 1;

void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR, dispL, dispR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbmL = cv::StereoBM::create(); 
  sbmL->setBlockSize(21);
  sbmL->setNumDisparities(32);
  sbmL->compute(grayL, grayR, dispL);

  // auto wls_filter = cv::ximgproc::createDisparityWLSFilter(sbmL);
  // cv::Ptr<cv::StereoMatcher> sbmR = cv::ximgproc::createRightMatcher(sbmL);
  // sbmR->compute(grayR, grayL, dispR);
  // wls_filter->setLambda(8000);
  // wls_filter->setSigmaColor(0.8);
  // wls_filter->filter(dispL, imgL, dispL, dispR);
  disp = dispL/16;

  // cv::Mat disp8;
  // cv::normalize(dispL, disp, 0, 255, 32, CV_8U);
  // cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
  // cv::imshow("disp", imgL+imgR);
  // cv::waitKey(10);
}

void reconstruction(cv::Mat img, cv::Mat &Q, cv::Mat &disp, cv::Mat &rectified, cv::Mat &XYZ){
  // cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
  //   350.6847, 0, 332.4661,
  //   0, 350.0606, 163.7461,
  //   0, 0, 1);
  // cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  // cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
  //   351.9498, 0, 329.4456,
  //   0, 351.0426, 179.0179,
  //   0, 0, 1);
  // cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  // cv::Mat R = (cv::Mat_<double>(3, 3) <<
  //   0.9997, 0.0015, 0.0215,
  //   -0.0015, 1, -0.00008,
  //   -0.0215, 0.00004, 0.9997);
  // cv::Mat T = (cv::Mat_<double>(3, 1) << -0.1191807, 0.0001532, 0.0011225);
  // cv::Size stdSize = cv::Size(WIDTH, HEIGHT);

  //official
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    349.891, 0, 334.352,
    0, 349.891, 187.937,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.173042, 0.0258831, 0, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    350.112, 0, 345.88,
    0, 350.112, 189.891,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.174209, 0.026726, 0, 0, 0);
  cv::Mat rodrigues = (cv::Mat_<double>(3, 1) << -0.0132397, 0.021005, -0.00121284);
  cv::Mat R;
  cv::Rodrigues(rodrigues, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -0.12, 0, 0);
  cv::Size stdSize = cv::Size(WIDTH, HEIGHT);

  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  cv::resize(imgL, imgL, stdSize);
  cv::resize(imgR, imgR, stdSize);

  //std::cout << imgR.size() <<std::endl;

  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize, &validRoI[0], &validRoI[1]);

  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  // //check whether the camera is facing forward
  // cv::Mat rectify = imgL+imgR;
  // cv::line(rectify, cv::Point(336,0), cv::Point(336,378),cv::Scalar(0,0,0),1);
  // cv::namedWindow("rectified", cv::WINDOW_NORMAL);
  // cv::imshow("rectified", rectify);
  // cv::waitKey(0);

  // cv::imwrite("tmp/imgL.png", imgL);
  // cv::imwrite("tmp/imgR.png", imgR);
  // return;

  blockMatching(disp, imgL, imgR);

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", imgL+imgR);
  // cv::waitKey(0);
  // cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
}

void convertImage(cv::Mat img, int w, int h, tiny_dnn::vec_t &data){
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(w, h));
  data.resize(w * h * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
       data[c * w * h + y * w + x] =
         float(resized.at<cv::Vec3b>(y, x)[c] / 255.0);
      }
    }
  }
}

void slidingWindow(const std::string& dictionary) {
  using conv    = tiny_dnn::convolutional_layer;
  using pool    = tiny_dnn::max_pooling_layer;
  using fc      = tiny_dnn::fully_connected_layer;
  using tanh    = tiny_dnn::tanh_layer;
  using leaky_relu    = tiny_dnn::leaky_relu_layer;
  using softmax = tiny_dnn::softmax_layer;

  tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

  // m_slidingWindow << conv(25, 25, 4, 3, 16, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh() 
  //    // << dropout(22*22*16, 0.25)                    
  //    << pool(22, 22, 16, 2, backend_type)                               
  //    << conv(11, 11, 4, 16, 32, tiny_dnn::padding::valid, true, 1, 1, backend_type) << tanh() 
  //    // << dropout(8*8*32, 0.25)                    
  //    << pool(8, 8, 32, 2, backend_type) 
  //    << fc(4 * 4 * 32, 128, true, backend_type) << leaky_relu()  
  //    << fc(128, 5, true, backend_type) << softmax(5);

  // m_slidingWindow << conv(45, 45, 3, 3, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
  //    // << dropout(22*22*16, 0.25)                                                   
  //    << conv(22, 22, 4, 16, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
  //    // << dropout(10*10*32, 0.25)
  //    << conv(10, 10, 4, 32, 64, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
  //    // << dropout(4*4*64, 0.25)                     
  //    << fc(4 * 4 * 64, 128, true, backend_type) << leaky_relu()  
  //    << fc(128, 5, true, backend_type) << softmax(5); 

  m_slidingWindow << conv(64, 64, 4, 3, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(31*31*16, 0.25)                                                   
     << conv(31, 31, 3, 16, 16, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(15*15*16, 0.25)
     << conv(15, 15, 3, 16, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(7*7*32, 0.25)
     << conv(7, 7, 3, 32, 32, tiny_dnn::padding::valid, true, 2, 2, backend_type) << tanh() 
     // << dropout(3*3*32, 0.25)                     
     << fc(3 * 3 * 32, 128, true, backend_type) << leaky_relu()  
     << fc(128, 5, true, backend_type) << softmax(5); 

  // load nets
  std::ifstream ifs(dictionary.c_str());
  ifs >> m_slidingWindow;
}

std::vector <cv::Point> imRegionalMax(cv::Mat input, int nLocMax, double threshold, int minDistBtwLocMax)
{
    cv::Mat scratch = input.clone();
    // std::cout<<scratch<<std::endl;
    // cv::GaussianBlur(scratch, scratch, cv::Size(3,3), 0, 0);
    std::vector <cv::Point> locations(0);
    locations.reserve(nLocMax); // Reserve place for fast access
    for (int i = 0; i < nLocMax; i++) {
        cv::Point location;
        double maxVal;
        cv::minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
        if (maxVal > threshold) {
            int col = location.x;
            int row = location.y;
            locations.push_back(cv::Point(col, row));
            int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
            int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
            int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
            int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
            for (int r = r0; r <= r1; r++) {
                for (int c = c0; c <= c1; c++) {
                    if (sqrt((r-row)*(r-row)+(c-col)*(c-col)) <= minDistBtwLocMax) {
                      scratch.at<double>(r,c) = 0.0;
                    }
                }
            }
        } else {
            break;
        }
    }
    return locations;
}


float median(std::vector<float> vec) {
  int size = vec.size();
  float tvecan;
  if (size % 2 == 0) { // even
    tvecan = (vec[vec.size() / 2 - 1] + vec[vec.size() / 2]) / 2;
  }

  else //odd
    tvecan = vec[vec.size() / 2];
  return (tvecan);
}

float_t depth2resizeRate(double x, double y){
  return 2*(1.6078-0.4785*std::sqrt(std::sqrt(x*x+y*y)));
}

struct Pt
{
  cv::Point2d pt;
  int group;
};

void gather_points(//初始化
  cv::Mat source,
  std::vector<float> vecQuery,
  std::vector<int> &vecIndex,
  std::vector<float> &vecDist
  )
{  
  double radius = 1;
  unsigned int max_neighbours = 100;
  cv::flann::KDTreeIndexParams indexParams(2);
  cv::flann::Index kdtree(source, indexParams); //此部分建立kd-tree索引同上例，故不做详细叙述
  cv::flann::SearchParams params(1024);//设置knnSearch搜索参数
  kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);
}


bool setRange(int x)

{
    return (x != 0);
}

bool compareGroup(Pt a)

{
    return (a.group == -1);
}

void filterKeypoints(std::vector<cv::Point3f>& point3Ds){
  std::vector<Pt> data;
  std::vector<cv::Point2f> data_tmp;
  
  for(int i = 0; i < point3Ds.size(); i++){
    if(point3Ds[i].y > 0.5 && point3Ds[i].y < 2){
      cv::Point2d pt(point3Ds[i].x, point3Ds[i].z);
      data_tmp.push_back(pt);
      data.push_back(Pt{pt,-1});
    }
  }
  point3Ds.clear();

  if(data.size() == 0)
    return;

  cv::Mat source = cv::Mat(data_tmp).reshape(1);
 
  int resultSize = 1000;
  double resultResize = 50;
  cv::RNG rng(time(0));
  cv::Mat result = cv::Mat::zeros(resultSize, resultSize, CV_8UC3);
  cv::Point2f point2D;
  int groupId = 0;
  

  for(int j = 0; j < data.size()-1; j++)
  {   
    if(data[j].group == -1){
      std::vector<float> vecQuery;//存放 查询点 的容器（本例都是vector类型）
      vecQuery.push_back(data[j].pt.x);
      vecQuery.push_back(data[j].pt.y);
      std::vector<int> vecIndex;
      std::vector<float> vecDist;

      gather_points(source, vecQuery, vecIndex, vecDist);//kd tree finish; find the points in the circle with point center vecQuery and radius, return index in vecIndex
      int num = std::count_if(vecIndex.begin(), vecIndex.end(), setRange);//if there is one lonely point, build it as an individual group
      for (int i = 1; i < vecIndex.size(); i++){
        if (vecIndex[i] == 0 && vecIndex[i+1] != 0){
          num++;
        }
      }
      if (num == 0){
        if (data[j].group == -1){ 
          data[j].group = groupId++;
          point2D = data[j].pt;
          // std::cout<<j<<" type 1"<<" "<<data[j].pt.x<<","<<data[j].pt.y<<" group "<<data[j].group<<std::endl;
        }
      }
      else{   
        std::vector<Pt> groupAll;
        std::vector<int> filteredIndex;
        std::vector<float> centerPointX;
        std::vector<float> centerPointY;
        for (int v = 0; v < num; v++){
          groupAll.push_back(data[vecIndex[v]]);
          filteredIndex.push_back(vecIndex[v]);
        }
      
        int noGroup = count_if(groupAll.begin(), groupAll.end(), compareGroup);
        if (noGroup > 0){
          for (int k = 0; k < filteredIndex.size(); k++)
          { 
            if (data[filteredIndex[k]].group == -1)
            { 
              data[filteredIndex[k]].group = groupId;
              centerPointX.push_back(data[vecIndex[k]].pt.x);
              centerPointY.push_back(data[vecIndex[k]].pt.y);

              float X1 = data[filteredIndex[k]].pt.x*resultResize+resultSize/2;
              float Y1 = data[filteredIndex[k]].pt.y*resultResize;
              // std::cout<<k<<" type 2"<<" "<<data[vecIndex[k]].pt.x<<","<<data[vecIndex[k]].pt.y<<" group "<< data[vecIndex[k]].group<<std::endl;
              cv::circle(result, cv::Point(X1,Y1), 5, cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)), -1);
            }
          }
          groupId++;
          point2D.x = median(centerPointX);
          point2D.y = median(centerPointY);
        }
        else{
          data[j].group = data[vecIndex[0]].group;
          point2D = data[j].pt;
          // std::cout<<j<<" type 2"<<" "<<data[j].pt.x<<","<<data[j].pt.y<<" group "<<data[j].group<<std::endl;
        }
      }
      point3Ds.push_back(cv::Point3f(point2D.x, 1, point2D.y));

      float X1 = point2D.x*resultResize+resultSize/2;
      float Y1 = point2D.y*resultResize;
      cv::circle(result, cv::Point(X1,Y1), 8, cv::Scalar(0, 255, 255), -1);
    }
  }

  // for (int p = 0; p < data.size(); p++){
  //   std::cout<<p<<" "<<data[p].pt.x<<" "<<data[p].pt.y<<" group "<<data[p].group<<std::endl;
  // }

  // for (int r = 0; r < point3Ds.size(); r++){
  //   std::cout<<"NO."<<r<<" "<<point3Ds[r].x<<","<<point3Ds[r].z<<std::endl;
  // }

  // cv::flip(result, result, 0);
  // cv::namedWindow("result", cv::WINDOW_NORMAL);
  // cv::imshow("result", result);
  // cv::waitKey(0);
}

void xyz2xy(cv::Mat Q, cv::Point3f xyz, cv::Point2f &xy, int &radius){
  double X = xyz.x;
  double Y = xyz.y;
  double Z = xyz.z;
  double Cx = -Q.at<double>(0,3);
  double Cy = -Q.at<double>(1,3);
  double f = Q.at<double>(2,3);
  double a = Q.at<double>(3,2);
  double b = Q.at<double>(3,3);
  double d = (f - Z * b ) / ( Z * a);
  xy.x = X * ( d * a + b ) + Cx;
  xy.y = Y * ( d * a + b ) + Cy;
  radius = int(0.4 * ( d * a + b ));
}

int countFiles(const char* path){
  struct dirent *de;
  DIR *dir = opendir(path);
  if(!dir)
  {
      printf("opendir() failed! Does it exist?\n");
      return -1;
  }

  unsigned long count = 0;
  while(de = readdir(dir))
  {
      ++count;
  }

  closedir(dir);
  return count;
}

void forwardDetectionORB(const std::string& imgPath){
  //Given RoI by SIFT detector and detected by CNN
  double threshold = 0.1;

  std::vector<tiny_dnn::tensor_t> inputs;
  std::vector<int> verifiedIndex;
  std::vector<cv::Point> candidates;
  // std::vector<int> outputs;

  cv::Mat img = cv::imread(imgPath);
  cv::Mat Q, disp, XYZ, imgRoI, imgSource;
  reconstruction(img, Q, disp, img, XYZ);
  img.copyTo(imgSource);

  // int rowT = 160;
  // int rowB = 290;
  int rowT = 190;
  int rowB = 320;
  // int rowT = 180;
  // int rowB = 376;
  imgRoI = img.rowRange(rowT, rowB);

  cv::Ptr<cv::ORB> detector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(imgRoI, keypoints);

  // cv::Mat Match;
  // cv::drawKeypoints(gray, keypoints, Match);
  // cv::namedWindow("Match", cv::WINDOW_NORMAL);
  // cv::imshow("Match", Match);
  // cv::waitKey(0);

  // cv::resize(img, img, cv::Size(WIDTH/2, HEIGHT/2));
  // cv::Mat probMap = cv::Mat::zeros(HEIGHT/2, WIDTH/2, CV_64F);
  // cv::Mat indexMap = cv::Mat::zeros(HEIGHT/2, WIDTH/2, CV_32S);
  cv::Mat probMap = cv::Mat::zeros(HEIGHT, WIDTH, CV_64F);
  cv::Mat indexMap = cv::Mat::zeros(HEIGHT, WIDTH, CV_32S);

  std::vector<cv::Point3f> point3Ds;
  cv::Point2f point2D;
  for(size_t i = 0; i < keypoints.size(); i++){
    cv::Point position(keypoints[i].pt.x, keypoints[i].pt.y+rowT);
    point3Ds.push_back(cv::Point3f(XYZ.at<cv::Point3f>(position)));
    // std::cout << cv::Point3f(XYZ.at<cv::Point3f>(position)) << std::endl;
  }
  filterKeypoints(point3Ds);
  for(size_t i = 0; i < point3Ds.size(); i++){
    int radius;
    xyz2xy(Q, point3Ds[i], point2D, radius);
    int x = point2D.x;
    int y = point2D.y;

    // float_t ratio = depth2resizeRate(point3Ds[i].x, point3Ds[i].z);
    // int length = ratio * 25;
    // int radius = (length-1)/2;
    // radius = 12;

    cv::Rect roi;
    roi.x = std::max(x - radius, 0);
    roi.y = std::max(y - radius, 0);
    roi.width = std::min(x + radius, img.cols) - roi.x;
    roi.height = std::min(y + radius, img.rows) - roi.y;

    //cv::circle(img, cv::Point (x,y), radius, cv::Scalar (0,0,0));
    // // cv::circle(disp, cv::Point (x,y), 3, 0, CV_FILLED);
    // cv::namedWindow("roi", cv::WINDOW_NORMAL);
    // cv::imshow("roi", img_hsv);
    // cv::waitKey(0);
    //cv::destroyAllWindows();

    if (0 > roi.x || 0 > roi.width || roi.x + roi.width > img.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > img.rows){
      std::cout << "Wrong roi!" << std::endl;
      // outputs.push_back(-1);
    }
    else{
      auto patchImg = img(roi);
      // cv::namedWindow("roi", cv::WINDOW_NORMAL);
      // cv::imshow("roi", patchImg);
      // cv::waitKey(0);
      tiny_dnn::vec_t data;
      convertImage(patchImg, PATCH_SIZE, PATCH_SIZE, data);
      inputs.push_back({data});
      // outputs.push_back(0);
      verifiedIndex.push_back(i);
      candidates.push_back(cv::Point(x,y));
    }
  }
  
  int index, index2;
  std::string filename, savePath;
  index = imgPath.find_last_of('/');
  filename = imgPath.substr(index+1);
  index2 = filename.find_last_of('.');
  std::ofstream savefile;
  savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".csv";
  savefile.open(savePath);

  int resultWidth = WIDTH;
  int resultHeight = 600;
  double resultResize = 30;
  cv::Mat result = cv::Mat::zeros(resultHeight, resultWidth, CV_8UC3);
  std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};

  if(inputs.size()>0){
    auto prob = m_slidingWindow.predict(inputs);
    for(size_t i = 0; i < inputs.size(); i++){
      size_t maxIndex = 0;
      double maxProb = prob[i][0][0];
      for(size_t j = 1; j < 5; j++){
        if(prob[i][0][j] > maxProb){
          maxIndex = j;
          maxProb = prob[i][0][j];
        }
      }
      // outputs[verifiedIndex[i]] = maxIndex;
      int x = candidates[i].x;
      int y = candidates[i].y;
      probMap.at<double>(y,x) = maxProb;
      indexMap.at<int>(y,x) = maxIndex;
    }
    std::vector <cv::Point> cones = imRegionalMax(probMap, 10, threshold, 10);

    for(size_t i = 0; i < cones.size(); i++){
      int x = cones[i].x;
      int y = cones[i].y;
      double maxProb = probMap.at<double>(y,x);
      int maxIndex = indexMap.at<int>(y,x);
      cv::Point position(x, y);
      cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
      std::string labelName = labels[maxIndex];
      // float_t ratio = depth2resizeRate(point3D.x, point3D.z);
      // int length = ratio * 25;
      // int radius = (length-1)/2;
      int radius;
      cv::Point2f position_tmp;
      xyz2xy(Q, point3D, position_tmp, radius);

      if(ANNOTATE){
        int rand = std::rand()%10;
        std::string path;
        if(rand < 7){
          path = "tmp/data/train/";
        }
        else{
          path = "tmp/data/test/";
        }
        path += std::to_string(maxIndex);
        int num = countFiles(path.c_str());
        path += "/"+std::to_string(maxIndex)+"_"+std::to_string(num)+".png";

        cv::Rect roi;
        roi.x = std::max(position.x - radius, 0);
        roi.y = std::max(position.y - radius, 0);
        roi.width = std::min(position.x + radius, img.cols) - roi.x;
        roi.height = std::min(position.y + radius, img.rows) - roi.y;
        if (0 > roi.x || 0 > roi.width || roi.x + roi.width > img.cols || 0 > roi.y || 0 > roi.height || roi.y + roi.height > img.rows){
          continue;
        }
        cv::Mat image = imgSource(roi);
        cv::resize(image, image, cv::Size(PATCH_SIZE,PATCH_SIZE));
        cv::imwrite(path, image);
      }

      if(radius>0){
        if (labelName == "background"){
          std::cout << "No cone detected" << std::endl;
          cv::circle(img, position, radius, cv::Scalar (0,0,0));
        } 
        else{
          if (labelName == "blue")
            cv::circle(img, position, radius, cv::Scalar (175,238,238));
          else if (labelName == "yellow")
            cv::circle(img, position, radius, cv::Scalar (0,255,255));
          else if (labelName == "orange")
            cv::circle(img, position, radius, cv::Scalar (0,165,255));
          else if (labelName == "big orange")
            cv::circle(img, position, radius, cv::Scalar (0,0,255));

          int xt = int(point3D.x * float(resultResize) + resultWidth/2);
          int yt = int(point3D.z * float(resultResize));
          if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
            if (labelName == "blue")
              cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
            else if (labelName == "yellow")
              cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
            else if (labelName == "orange")
              cv::circle(result, cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
            else if (labelName == "big orange")
              cv::circle(result, cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
          }

          std::cout << position << " " << labelName << " " << point3D << " " << maxProb << std::endl;
          savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+labelName+","+std::to_string(point3D.x)+","+std::to_string(point3D.y)+","+std::to_string(point3D.z)+"\n";
        }
      }
    }
  }
      

  for(int i = 0; i < keypoints.size(); i++){
    cv::circle(img, cv::Point(keypoints[i].pt.x,keypoints[i].pt.y+rowT), 2, cv::Scalar (255,255,255), -1);
  }

  cv::line(img, cv::Point(0,rowT), cv::Point(WIDTH,rowT), cv::Scalar(0,0,255), 2);
  cv::line(img, cv::Point(0,rowB), cv::Point(WIDTH,rowB), cv::Scalar(0,0,255), 2);

  // int resultWidth = 672;
  // int resultHeight = 600;
  // double resultResize = 30;
  // cv::Mat result[2] = cv::Mat::zeros(resultHeight, resultWidth, CV_8UC3), coResult;
  // std::string labels[] = {"background", "blue", "yellow", "orange", "big orange"};
  // if(inputs.size()>0){
  //   auto prob = m_slidingWindow.predict(inputs);
  //   for(size_t i = 0; i < inputs.size(); i++){
  //     size_t maxIndex = 1;
  //     double maxProb = prob[i][0][1];
  //     for(size_t j = 2; j < 5; j++){
  //       if(prob[i][0][j] > maxProb){
  //         maxIndex = j;
  //         maxProb = prob[i][0][j];
  //       }
  //     }
  //     // outputs[verifiedIndex[i]] = maxIndex;
  //     int x = candidates[i].x;
  //     int y = candidates[i].y;
  //     cv::Point position(x*2, y*2+180);
  //     cv::Point3f point3D = XYZ.at<cv::Point3f>(position);
  //     std::string labelName = labels[maxIndex];     

  //     if (labelName == "background"){
  //       std::cout << "No cone detected" << std::endl;
  //       cv::circle(img, position, 2, cv::Scalar (0,0,0), -1);
  //     } 
  //     else{
  //       // std::cout << "Find one " << labelName << " cone"<< std::endl;
  //       if (labelName == "blue")
  //         cv::circle(img, position, 2, cv::Scalar (175,238,238), -1);
  //       else if (labelName == "yellow")
  //         cv::circle(img, position, 2, cv::Scalar (0,255,255), -1);
  //       else if (labelName == "orange")
  //         cv::circle(img, position, 2, cv::Scalar (0,165,255), -1);
  //       else if (labelName == "big orange")
  //         cv::circle(img, position, 4, cv::Scalar (0,0,255), -1);

  //       int xt = int(point3D.x * float(resultResize) + resultWidth/2);
  //       int yt = int((point3D.z-1.872f) * float(resultResize));
  //       if (xt >= 0 && xt <= resultWidth && yt >= 0 && yt <= resultHeight){
  //         if (labelName == "blue")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (255,0,0), -1);
  //         else if (labelName == "yellow")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,255,255), -1);
  //         else if (labelName == "orange")
  //           cv::circle(result[0], cv::Point (xt,yt), 5, cv::Scalar (0,165,255), -1);
  //         else if (labelName == "big orange")
  //           cv::circle(result[0], cv::Point (xt,yt), 10, cv::Scalar (0,0,255), -1);
  //       }

  //       std::cout << position << " " << labelName << " " << point3D << std::endl;
  //       savefile << std::to_string(position.x)+","+std::to_string(position.y)+","+labelName+","+std::to_string(point3D.x)+","+std::to_string(point3D.y)+","+std::to_string(point3D.z)+"\n";
  //     }
  //   }
  // }


  // for(int i = 0; i < m_finalPointCloud.cols(); i++){
  //   savefile << std::to_string(m_finalPointCloud(0,i))+","+std::to_string(m_finalPointCloud(1,i))+","+std::to_string(m_finalPointCloud(2,i))+"\n";
  //   int x = int(m_finalPointCloud(0,i) * resultResize + resultWidth/2);
  //   int y = int(m_finalPointCloud(1,i) * resultResize);
  //   if (x >= 0 && x <= resultWidth && y >= 0 && y <= resultHeight){
  //     cv::circle(result[0], cv::Point (x,y), 5, cv::Scalar (255, 255, 255), -1);
  //   }
  // }

  // cv::circle(result[0], cv::Point (int(resultWidth/2),0), 5, cv::Scalar (0, 0, 255), -1);
  cv::flip(result, result, 0);
  // img.copyTo(result[1].rowRange(resultHeight-376,resultHeight));
  // cv::hconcat(result[1], result[0], coResult);
  cv::imwrite(imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".png", img);

  // savePath = imgPath.substr(0,index-7)+"/results/"+filename.substr(0,index2)+".png";
  // cv::imwrite(savePath, img);

  // savePath = imgPath.substr(0,index-7)+"/disp_filtered/"+filename.substr(0,index2)+".png";
  // std::cout<<savePath<<std::endl;
  // cv::imwrite(savePath, disp);

  cv::namedWindow("img", cv::WINDOW_NORMAL);
  cv::imshow("img", img);
  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", disp);
  cv::waitKey(30);
  // cv::destroyAllWindows();

  // for(size_t i = 0; i < pts.size(); i++)
  //   std::cout << i << ": " << outputs[i] << std::endl;
}






int main( int argc, char** argv )
{
  /*if(ANNOTATE){
    boost::filesystem::remove_all("tmp/data");
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/train/0"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/train/1"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/train/2"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/train/3"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/train/4"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/test/0"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/test/1"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/test/2"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/test/3"));
    boost::filesystem::create_directories(boost::filesystem::path("/tmp/data/test/4"));
  }*/
	slidingWindow("models/all_roi_big");
  // forwardDetectionORB("annotations/circle/images/190.png");
	for(int i = 85; i < 650; i++){
		auto startTime = std::chrono::system_clock::now();
		forwardDetectionORB("annotations/circle/images/"+std::to_string(i)+".png");
		
		auto endTime = std::chrono::system_clock::now();
  	std::chrono::duration<double> diff = endTime-startTime;
  	std::cout << "Time: " << diff.count() << " s\n";
	}
}
