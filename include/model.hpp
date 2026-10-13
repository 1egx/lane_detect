#ifndef LIBLDW_TWINLITENET_DNN_H_
#define LIBLDW_TWINLITENET_DNN_H_

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class UFLDNet {
public:
    UFLDNet(const std::string& model_path, bool use_cuda);
    std::vector<std::vector<cv::Point>> Infer(const cv::Mat& frame);
    double GetLastInferenceTimeMs() const;
    bool IsUsingCuda() const;

private:
    void SetCpuBackend(bool reload_model);
    std::vector<std::vector<cv::Point>> Decode(
        const std::vector<cv::Mat>& outputs, int image_width, int image_height) const;

    cv::dnn::Net net_;
    std::string model_path_;
    bool using_cuda_ = false;
    double last_inference_time_ms_ = 0.0;
};

#endif