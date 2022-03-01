#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class SfmLeaner {
private:
    cv::dnn::Net net;
    const int input_width = 480;
    const int input_height = 960;
public:
    SfmLeaner(std::string model_name);
    void infer(cv::Mat& frame);
};

