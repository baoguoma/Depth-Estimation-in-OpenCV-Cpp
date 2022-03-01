#include "SfmLeaner.h"
using namespace cv;
using namespace dnn;
using namespace std;

SfmLeaner::SfmLeaner(std::string model_name) {
    std::cout << "Model name:" << model_name << std::endl;
    auto model_file = model_name + ".onnx";
    net = readNet(model_file);
}

void SfmLeaner::infer (Mat& frame) {
    Mat blob;
    blobFromImage(frame, blob, 2 , Size(input_width, input_height), 
                  Scalar(0.5, 0.5, 0.5), true, false);
    net.setInput(blob);
    Mat output_blob;
    vector<Mat> output_img;
//	net.forward(output, this->net.getUnconnectedOutLayersNames());
    output_blob = net.forward();
    imagesFromBlob(output_blob, output_img);
    imshow("SfmLearner",output_img[0]);
    waitKey(0);
}

int main() {
    SfmLeaner test("depth_estimation");
    string img_path = "bus.jpg";
    Mat src_img = imread(img_path);
    
    auto start = chrono::steady_clock::now();
    test.infer(src_img);
    auto end = chrono::steady_clock::now();
    double ratio = (double) chrono::steady_clock::duration::period::num/
                            chrono::steady_clock::duration::period::den;
    
    cout << (end - start).count() * ratio;
    return 0;
}