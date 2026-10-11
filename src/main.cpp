#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "twinlitenet_dnn.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }

    TwinLiteNet twinlitenet("/home/xsjhh/Desktop/TwinLiteNet-onnx-opencv-dnn-main/models/best.onnx");
    std::cout << "Model Loaded" << std::endl;

    cv::VideoCapture capture(argv[1]);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open video: " << argv[1] << std::endl;
        return 1;
    }

    const double fps = capture.get(cv::CAP_PROP_FPS);
    const int frame_delay = fps > 0.0 ? std::max(1, static_cast<int>(1000.0 / fps)) : 1;
    cv::namedWindow("TwinLiteNet Video", cv::WINDOW_NORMAL);

    cv::Mat frame;
    while (capture.read(frame)) {
        cv::Mat model_input;
        cv::resize(frame, model_input, cv::Size(640, 360));

        cv::Mat da_out, ll_out;
        twinlitenet.Infer(model_input, da_out, ll_out);

        cv::Mat da_mask, ll_mask;
        cv::resize(da_out, da_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(ll_out, ll_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
        frame.setTo(cv::Scalar(255, 0, 127), da_mask);
        frame.setTo(cv::Scalar(0, 0, 255), ll_mask);

        cv::imshow("TwinLiteNet Video", frame);
        const int key = cv::waitKey(frame_delay);
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
