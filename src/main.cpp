#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "model.hpp"
#include "tradition.hpp"

struct ProgramOptions {
    std::string video_path;
    std::string device = "GPU";
    std::string method = "model";
};


void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name
              << " <video_path> [--device <GPU|CPU>] [--method <model|tradition|both>]\n"
              << "       " << program_name << " --help" << std::endl;
}

void parseArguments(int argc, char* argv[], ProgramOptions& options) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(0);
    }

    if (argc < 2 || argv[1][0] == '-') {
        std::cerr << "Error: the first argument must be a video path" << std::endl;
        printUsage(argv[0]);
        exit(1);
    }
    options.video_path = argv[1];

    bool device_set = false;
    for (int index = 2; index < argc; index += 2) {
        const std::string option = argv[index];
        if (index + 1 >= argc || argv[index + 1][0] == '-') {
            std::cerr << "Error: missing value for " << option << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
        if(option == "--device") {
            if (device_set) {
                std::cerr << "Error: --device was specified more than once" << std::endl;
                exit(1);
            }
            options.device = argv[index + 1];
            if (options.device != "GPU" && options.device != "CPU") {
                std::cerr << "Error: device must be 'GPU' or 'CPU'" << std::endl;
                exit(1);
            }
            device_set = true;
        } else if(option == "--method") {
            options.method = std::string(argv[index + 1]);
            if(options.method != "model" && options.method != "tradition" && options.method != "both") {
                std::cerr << "Error: --method must be 'model' , 'tradition' or 'both' " << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Error: unknown option " << option << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
    }

}

void model_print(const std::vector<std::vector<cv::Point>>& lanes,const cv::Mat frame,const double time) {
    for (const auto& lane : lanes) {
            for (const auto& point : lane) {
                cv::circle(frame, point, 5, cv::Scalar(0, 255, 0), -1);
            }
        }

        const std::string timing_text = cv::format("Inference: %.2f ms", time);
        cv::putText(frame, timing_text, cv::Point(18, 38), cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(frame, timing_text, cv::Point(18, 38), cv::FONT_HERSHEY_SIMPLEX,
                    0.9, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::imshow("Lane Detection - Model", frame);
}

void tradition_print(const cv::Mat frame, const double time) {
    const std::string timing_text = cv::format("Inference: %.2f ms", time);
    cv::putText(frame, timing_text, cv::Point(18, 38), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, timing_text, cv::Point(18, 38), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    cv::imshow("Lane Detection - Tradition", frame);
}

int main(int argc, char* argv[]) {
    //加载函数
    ProgramOptions options;
    parseArguments(argc, argv, options);

    //模型配置
    UFLDNet model("models/curvelanes_res34.onnx", options.device == "GPU");
    std::cout << "Model Loaded (" << (model.IsUsingCuda() ? "GPU" : "CPU") << ")" << std::endl;

    cv::VideoCapture capture(options.video_path);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open video: " << options.video_path << std::endl;
        return 1;
    }

    Tradition tradition;

    cv::Mat frame;
    while (capture.read(frame)) {
        if(options.method == "model" || options.method == "both") {
            const cv::Mat model_frame = frame.clone();
            const auto lanes = model.Infer(model_frame);
            model_print(lanes, model_frame, model.GetLastInferenceTimeMs());
        }
        if(options.method == "tradition" || options.method == "both") {
            const cv::Mat tradition_frame = frame.clone();
            tradition_print(tradition.Infer(tradition_frame), tradition.GetLastInferenceTimeMs());
        }
        const int key = cv::pollKey();
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}
