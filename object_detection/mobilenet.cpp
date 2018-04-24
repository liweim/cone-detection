#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dict.hpp>
#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

// #include <opencv2\opencv.hpp>
// #include <opencv2\dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;

const size_t inWidth = 300;
const size_t inHeight = 300;
// const float WHRatio = inWidth / (float)inHeight;
const char* classNames[] = { "background", "blue", "yellow", "orange", "orange2" };

void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR;

  cv::cvtColor(imgL, grayL, 6);
  cv::cvtColor(imgR, grayR, 6);

  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(); 
  sbm->setBlockSize(17);
  sbm->setNumDisparities(32);

  sbm->compute(grayL, grayR, disp);
  cv::normalize(disp, disp, 0, 255, 32, CV_8U);
}

void reconstruction(cv::Mat img, cv::Mat &Q, cv::Mat &disp, cv::Mat &rectified, cv::Mat &XYZ){
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    350.6847, 0, 332.4661,
    0, 350.0606, 163.7461,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    351.9498, 0, 329.4456,
    0, 351.0426, 179.0179,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  cv::Mat R = (cv::Mat_<double>(3, 3) <<
    0.9997, 0.0015, 0.0215,
    -0.0015, 1, -0.00008,
    -0.0215, 0.00004, 0.9997);
  //cv::transpose(R, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);

  cv::Size stdSize = cv::Size(640, 360);
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

  //cv::imwrite("2_left.png", imgL);
  //cv::imwrite("2_right.png", imgR);

  blockMatching(disp, imgL, imgR);

  // cv::namedWindow("disp", cv::WINDOW_NORMAL);
  // cv::imshow("disp", disp);
  // cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
  XYZ *= 0.002;
}

void object_detection(String imgPath, float confidenceThreshold) {
    String weights = "frozen_inference_graph.pb";
    String prototxt = "ssd_mobilenet_v1_coco.pbtxt";
    dnn::Net net = cv::dnn::readNetFromTensorflow(weights, prototxt);

    Mat imgSource = cv::imread(imgPath);
    cv::Mat Q, disp, rectified, XYZ, img;
    reconstruction(imgSource, Q, disp, rectified, XYZ);

    cv::resize(rectified, rectified, cv::Size(inWidth, inHeight));
    // rectified.rowRange(0,150) = 0;
    // Size rectified_size = rectified.size();

    // Size cropSize;
    // if (rectified_size.width / (float)rectified_size.height > WHRatio)
    // {
    //     cropSize = Size(static_cast<int>(rectified_size.height * WHRatio),
    //         rectified_size.height);
    // }
    // else
    // {
    //     cropSize = Size(rectified_size.width,
    //         static_cast<int>(rectified_size.width / WHRatio));
    // }

    // Rect crop(Point((rectified_size.width - cropSize.width) / 2,
    //     (rectified_size.height - cropSize.height) / 2),
    //     cropSize);


    cv::Mat blob = cv::dnn::blobFromImage(rectified,1./255);
    //cout << "blob size: " << blob.size << endl;

    net.setInput(blob);
    Mat output = net.forward();
    //cout << "output size: " << output.size << endl;

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // rectified = rectified(crop);
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * rectified.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * rectified.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * rectified.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * rectified.rows);

            ostringstream ss;
            ss << confidence;
            String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

            if(objectClass == 1)
                rectangle(rectified, object, Scalar(255, 0, 0),1);
            else if(objectClass == 2)
                rectangle(rectified, object, Scalar(0, 255, 255),1);
            else if(objectClass == 3)
                rectangle(rectified, object, Scalar(0, 165, 255),1);
            else if(objectClass == 4)
                rectangle(rectified, object, Scalar(0, 0, 255),2);
            // String label = String(classNames[objectClass]) + ": " + conf;
            // int baseLine = 0;
            // Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            // rectangle(rectified, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
            //     Size(labelSize.width, labelSize.height + baseLine)),
            //     Scalar(0, 255, 0), -1);
            // putText(rectified, label, Point(xLeftBottom, yLeftBottom),
            //     1, 0.5, Scalar(0, 0, 0));
        }
    }

    cv::resize(rectified, rectified, cv::Size(640,360));

    std::string filename, savePath;
    int index = imgPath.find_last_of('/');
    filename = imgPath.substr(index+1);
    savePath = imgPath.substr(0,index-7)+"/results/"+filename;
    cv::imwrite(savePath, rectified);

    // namedWindow("image", cv::WINDOW_NORMAL);
    // imshow("image", rectified);
    // waitKey(0);
}

int main(int argc, char **argv){
    boost::filesystem::path dpath(argv[1]);
    BOOST_FOREACH(const boost::filesystem::path& imgPath, std::make_pair(boost::filesystem::directory_iterator(dpath), boost::filesystem::directory_iterator())) {
        std::cout << imgPath.string() << std::endl;

        auto startTime = std::chrono::system_clock::now();
        object_detection(imgPath.string(), stof(argv[2]));
        auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endTime-startTime;
        std::cout << "Time: " << diff.count() << " s\n";
    }
}