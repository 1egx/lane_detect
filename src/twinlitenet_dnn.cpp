#include "twinlitenet_dnn.hpp"

TwinLiteNet::TwinLiteNet(std::string model_path) : model_path_(model_path) {

    net_ = cv::dnn::readNetFromONNX(model_path_);

#ifdef ENABLE_CUDA
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#endif
}

void TwinLiteNet::Infer(const cv::Mat &image, cv::Mat &da_out, cv::Mat &ll_out) {
    assert(image.rows == 360 && image.cols == 640 && image.channels() == 3);

    cv::Mat input_img = image.clone();

    cv::Mat blob = cv::dnn::blobFromImage(input_img, 1.0/255.0,
                                          cv::Size(640, 360),
                                          cv::Scalar(), true, false);

    net_.setInput(blob);

    std::vector<cv::Mat> results;
    net_.forward(results, kModelOutputNames);

    cv::Mat x1, x2;
    results[0].col(1).reshape(0, 360).convertTo(x1, CV_8UC1, 255.0);
    results[1].col(1).reshape(0, 360).convertTo(x2, CV_8UC1, 255.0);

    da_out = x1.clone();
    ll_out = x2.clone();
}