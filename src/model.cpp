#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
constexpr int kInputWidth = 1600;
constexpr int kInputHeight = 800;
constexpr int kResizeHeight = 1000;
constexpr int kRowGridSize = 200;
constexpr int kRowCount = 72;
constexpr int kColGridSize = 100;
constexpr int kColCount = 81;
constexpr int kLaneCount = 10;

float tensorValue(const cv::Mat& tensor, int first, int second, int third) {
    const float* values = tensor.ptr<float>();
    const size_t offset =
        (static_cast<size_t>(first) * tensor.size[2] + second) * tensor.size[3] + third;
    return values[offset];
}

int argmax(const cv::Mat& tensor, int grid_size, int anchor, int lane) {
    int best_index = 0;
    float best_value = tensorValue(tensor, 0, anchor, lane);
    for (int index = 1; index < grid_size; ++index) {
        const float value = tensorValue(tensor, index, anchor, lane);
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

bool laneExists(const cv::Mat& tensor, int anchor, int lane) {
    return tensorValue(tensor, 1, anchor, lane) > tensorValue(tensor, 0, anchor, lane);
}

float expectedGridPosition(const cv::Mat& tensor, int grid_size, int anchor, int lane) {
    const int peak = argmax(tensor, grid_size, anchor, lane);
    const int first = std::max(0, peak - 1);
    const int last = std::min(grid_size - 1, peak + 1);
    float max_logit = tensorValue(tensor, peak, anchor, lane);
    for (int index = first; index <= last; ++index) {
        max_logit = std::max(max_logit, tensorValue(tensor, index, anchor, lane));
    }

    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int index = first; index <= last; ++index) {
        const float probability = std::exp(tensorValue(tensor, index, anchor, lane) - max_logit);
        denominator += probability;
        numerator += probability * index;
    }
    return numerator / denominator + 0.5f;
}
}  // namespace

UFLDNet::UFLDNet(const std::string& model_path, bool use_cuda) : model_path_(model_path) {
    net_ = cv::dnn::readNetFromONNX(model_path);

    if (!use_cuda) {
        SetCpuBackend(false);
        return;
    }

    try {
        const auto targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
        if (std::find(targets.begin(), targets.end(), cv::dnn::DNN_TARGET_CUDA) == targets.end()) {
            std::cerr << "CUDA target is unavailable; falling back to CPU." << std::endl;
            SetCpuBackend(false);
            return;
        }
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        using_cuda_ = true;
    } catch (const cv::Exception& error) {
        std::cerr << "CUDA initialization failed: " << error.what()
                  << "\nFalling back to CPU." << std::endl;
        SetCpuBackend(true);
    }
}

void UFLDNet::SetCpuBackend(bool reload_model) {
    if (reload_model) {
        net_ = cv::dnn::readNetFromONNX(model_path_);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    using_cuda_ = false;
}

std::vector<std::vector<cv::Point>> UFLDNet::Infer(const cv::Mat& frame) {
    if (frame.empty() || frame.channels() != 3) {
        throw std::invalid_argument("UFLDNet expects a non-empty 3-channel BGR frame");
    }

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(kInputWidth, kResizeHeight));
    const cv::Rect crop_rect(0, kResizeHeight - kInputHeight, kInputWidth, kInputHeight);
    cv::Mat rgb;
    cv::cvtColor(resized(crop_rect), rgb, cv::COLOR_BGR2RGB);

    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float standard_deviation[] = {0.229f, 0.224f, 0.225f};
    for (int channel = 0; channel < 3; ++channel) {
        channels[channel] = (channels[channel] - mean[channel]) / standard_deviation[channel];
    }
    cv::merge(channels, normalized);

    const cv::Mat input_blob = cv::dnn::blobFromImage(normalized);
    const std::vector<std::string> output_names = {
        "loc_row", "loc_col", "exist_row", "exist_col"};
    std::vector<cv::Mat> outputs;
    const int64 start_tick = cv::getTickCount();
    try {
        net_.setInput(input_blob);
        net_.forward(outputs, output_names);
    } catch (const cv::Exception& error) {
        if (!using_cuda_) {
            throw;
        }
        std::cerr << "CUDA inference failed: " << error.what()
                  << "\nRetrying this frame on CPU." << std::endl;
        SetCpuBackend(true);
        net_.setInput(input_blob);
        net_.forward(outputs, output_names);
    }
    last_inference_time_ms_ =
        (cv::getTickCount() - start_tick) * 1000.0 / cv::getTickFrequency();
    return Decode(outputs, frame.cols, frame.rows);
}

double UFLDNet::GetLastInferenceTimeMs() const {
    return last_inference_time_ms_;
}

bool UFLDNet::IsUsingCuda() const {
    return using_cuda_;
}

std::vector<std::vector<cv::Point>> UFLDNet::Decode(
    const std::vector<cv::Mat>& outputs, int image_width, int image_height) const {
    if (outputs.size() != 4) {
        throw std::runtime_error("Expected four UFLD output tensors");
    }
    const cv::Mat& loc_row = outputs[0];
    const cv::Mat& loc_col = outputs[1];
    const cv::Mat& exist_row = outputs[2];
    const cv::Mat& exist_col = outputs[3];
    if (loc_row.dims != 4 || loc_col.dims != 4 || exist_row.dims != 4 || exist_col.dims != 4 ||
        loc_row.size[1] != kRowGridSize || loc_row.size[2] != kRowCount ||
        loc_col.size[1] != kColGridSize || loc_col.size[2] != kColCount) {
        throw std::runtime_error("Unexpected UFLD output tensor dimensions");
    }

    std::vector<std::vector<cv::Point>> lanes(kLaneCount);
    for (int lane = 0; lane < kLaneCount; ++lane) {
        int valid_count = 0;
        for (int anchor = 0; anchor < kRowCount; ++anchor) {
            if (laneExists(exist_row, anchor, lane)) {
                ++valid_count;
                const float x = expectedGridPosition(loc_row, kRowGridSize, anchor, lane) /
                                (kRowGridSize - 1) * image_width;
                const float row_anchor = 0.4f + 0.6f * anchor / (kRowCount - 1);
                lanes[lane].emplace_back(cvRound(x), cvRound(row_anchor * image_height));
            }
        }
        if (valid_count <= kRowCount / 4) {
            lanes[lane].clear();
        }
    }

    for (int lane = 0; lane < kLaneCount; ++lane) {
        int valid_count = 0;
        for (int anchor = 0; anchor < kColCount; ++anchor) {
            if (laneExists(exist_col, anchor, lane)) {
                ++valid_count;
                const float y = expectedGridPosition(loc_col, kColGridSize, anchor, lane) /
                                (kColGridSize - 1) * image_height;
                const float col_anchor = static_cast<float>(anchor) / (kColCount - 1);
                lanes[lane].emplace_back(cvRound(col_anchor * image_width), cvRound(y));
            }
        }
        if (valid_count <= kColCount / 4) {
            lanes[lane].clear();
        }
    }
    return lanes;
}