#include "tradition.hpp"

Mat Tradition::applyGrayscale(Mat source) {
    Mat dst;
    cvtColor(source, dst, COLOR_BGR2GRAY);

    return dst;
}

Mat Tradition::applyGaussianBlur(Mat source) {
    Mat dst;
    GaussianBlur(source, dst, Size(3, 3), 0);

    return dst;
}

Mat Tradition::applyCanny(Mat source) {
    Mat dst;
    Canny(source, dst, 50, 150);

    return dst;
}

Mat Tradition::filterColors(Mat source, bool isDayTime) {
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;

    std::vector< int > lowerWhite = {130, 130, 130};
    std::vector< int > upperWhite = {255, 255, 255};
    inRange(source, lowerWhite, upperWhite, whiteMask);
    bitwise_and(source, source, whiteImage, whiteMask);

    cvtColor(source, hsv, COLOR_BGR2HSV);
    std::vector< int > lowerYellow = {20, 100, 110}; 
    std::vector< int > upperYellow = {30, 180, 240};
    inRange(hsv, lowerYellow, upperYellow, yellowMask);
    bitwise_and(source, source, yellowImage, yellowMask);

    addWeighted(whiteImage, 1., yellowImage, 1., 0., whiteYellow);

    if (isDayTime == false) {   
        Mat grayMask, grayImage, grayAndWhite, dst;
        std::vector< int > lowerGray = {80, 80, 80};
        std::vector< int > upperGray = {130, 130, 130};
        inRange(source, lowerGray, upperGray, grayMask);
        bitwise_and(source, source, grayImage, grayMask);

        addWeighted(grayImage, 1., whiteYellow, 1., 0., dst);
        return dst;
    }
    
    return whiteYellow;
}

Mat Tradition::RegionOfInterest(Mat source) {
    
    float trapezoidBottomWidth = 1.0; 
    float trapezoidTopWidth = 0.07; 
    float trapezoidHeight = 0.5; 

    float smallBottomWidth = 0.45; 
    float smallTopWidth = 0.3; 
    float smallHeight = 1.0; 

    float bar = 0.97;

    std::vector<Point> pts;
    
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar));
    pts.push_back(cv::Point((source.cols * (1 - trapezoidTopWidth)) / 2, source.rows - source.rows * trapezoidHeight));
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth) ) / 2, source.rows - source.rows * trapezoidHeight));
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth)) / 2, source.rows * bar));
    
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar)); 
    pts.push_back(cv::Point((source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight)); 
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2, source.rows - source.rows * trapezoidHeight * smallHeight));
    pts.push_back(cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2, source.rows * bar));

    Mat mask = Mat::zeros(source.size(), source.type());
    fillPoly(mask, pts, Scalar(255,255,255));


    Mat maskedImage;
    bitwise_and(source, mask, maskedImage);

    return maskedImage;
}

Mat Tradition::drawLanes(Mat source, std::vector<Vec4i> lines) {
    if (lines.size() == 0) {
        return source;
    }

    bool drawRightLane = true;
    bool drawLeftLane = true;

    float slopeThreshold = 0.5;
    std::vector< float > slopes;
    std::vector< Vec4i > goodLines;

    for (int i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        double slope;

        if (l[2] - l[0] == 0) {
            slope = 999;
        } else  {
            slope = 1.0 * (l[3] - l[1]) / (l[2] - l[0]);
        }
        if (abs(slope) > 0.5) {
            slopes.push_back(slope);
            goodLines.push_back(l);
        }
    }

    std::vector< Vec4i > rightLines;
    std::vector< Vec4i > leftLines;
    int imgCenter = source.cols / 2;

    for (int i = 0; i < slopes.size(); i++){
        if (slopes[i] > 0 && goodLines[i][0] > imgCenter && goodLines[i][2] > imgCenter) {
            rightLines.push_back(goodLines[i]);
        }
        if (slopes[i] < 0 && goodLines[i][0] < imgCenter && goodLines[i][2] < imgCenter) {
            leftLines.push_back(goodLines[i]);
        }
    }

    std::vector< int > rightLinesX;
    std::vector< int > rightLinesY;
    double rightB1, rightB0;

    for (int i = 0; i < rightLines.size(); i++) {
        rightLinesX.push_back(rightLines[i][0]);
        rightLinesX.push_back(rightLines[i][2]); 
        rightLinesY.push_back(rightLines[i][1]); 
        rightLinesY.push_back(rightLines[i][3]); 
    }

    if (rightLinesX.size() > 0) {
        std::vector< double > coefRight = estimateCoefficients<int, double>(rightLinesX, rightLinesY); // y = b1x + b0
        rightB1 = coefRight[0];
        rightB0 = coefRight[1];
    } else {
        rightB1 = 1;
        rightB0 = 1;
        drawRightLane = false;
    }

    std::vector< int > leftLinesX;
    std::vector< int > leftLinesY;
    double leftB1, leftB0;

    for (int i = 0; i < leftLines.size(); i++) {
        leftLinesX.push_back(leftLines[i][0]); 
        leftLinesX.push_back(leftLines[i][2]);
        leftLinesY.push_back(leftLines[i][1]);
        leftLinesY.push_back(leftLines[i][3]);
    }

    if (leftLinesX.size() > 0) {
        std::vector< double > coefLeft = estimateCoefficients<int, double>(leftLinesX, leftLinesY); 
        leftB1 = coefLeft[0];
        leftB0 = coefLeft[1];
    } else {
        leftB1 = 1;
        leftB0 = 1;
        drawLeftLane = false;
    }


    int y1 = source.rows;


    int y2 = source.rows * (1 - 0.4); 

    int rightX1 = (y1 - rightB0) / rightB1; 
    int rightX2 = (y2 - rightB0) / rightB1; 

    int leftX1 = (y1 - leftB0) / leftB1;
    int leftX2 = (y2 - leftB0) / leftB1; 

    if (rightX2 < leftX2 || leftX2 > rightX2) {
        return source;
    }

    static bool isFilterInitialized = false;
    static Size previousFrameSize;
    static KalmanFilter laneFilter;

    if (previousFrameSize != source.size()) {
        isFilterInitialized = false;
        previousFrameSize = source.size();
    }

    if (drawRightLane && drawLeftLane) {
        if (!isFilterInitialized) {
            laneFilter.init(4, 4, 0, CV_32F);
            laneFilter.transitionMatrix = Mat::eye(4, 4, CV_32F);
            laneFilter.measurementMatrix = Mat::eye(4, 4, CV_32F);
            laneFilter.processNoiseCov = Mat::eye(4, 4, CV_32F) * 4.0f;
            laneFilter.measurementNoiseCov = Mat::eye(4, 4, CV_32F) * 25.0f;
            laneFilter.errorCovPost = Mat::eye(4, 4, CV_32F);
            laneFilter.statePost = (Mat_<float>(4, 1) << rightX1, rightX2, leftX1, leftX2);
            isFilterInitialized = true;
        }

        laneFilter.predict();
        Mat measurements = (Mat_<float>(4, 1) << rightX1, rightX2, leftX1, leftX2);
        Mat filteredEndpoints = laneFilter.correct(measurements);

        rightX1 = cvRound(filteredEndpoints.at<float>(0));
        rightX2 = cvRound(filteredEndpoints.at<float>(1));
        leftX1 = cvRound(filteredEndpoints.at<float>(2));
        leftX2 = cvRound(filteredEndpoints.at<float>(3));
    }


    if (drawRightLane == true && drawLeftLane == true) {
        line(source, Point(rightX1, y1), Point(rightX2, y2), Scalar(0, 255, 0), 7);
        line(source, Point(leftX1, y1), Point(leftX2, y2), Scalar(0, 255, 0), 7);
        return source; 
    }
    
    return source;
}

std::vector<Vec4i> Tradition::houghLines(Mat canny, Mat source, bool drawHough) {
    double rho = 2;
    double theta = 1 * M_PI / 180;
    int thresh = 15;
    double minLineLength = 10;
    double maxGapLength = 20;
   
    std::vector<Vec4i> linesP;
    HoughLinesP(canny, linesP, rho, theta, thresh, minLineLength, maxGapLength);

    if (drawHough == true) {
        for( size_t i = 0; i < linesP.size(); i++) {
            Vec4i l = linesP[i];
            line( source, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
        }
        imshow("Hough Lines", source);
        waitKey(0);
    }

    return linesP;
}

bool Tradition::isDayTime(Mat source) {

    Scalar s = mean(source);
   
    if (s[0] < 30 || s[1] < 33 && s[2] < 30) {
        return false;
    }

    return true;
}

Mat Tradition::Infer(Mat frame) {
    const int64 start_tick = cv::getTickCount();
    bool isDay = isDayTime(frame);

    Mat filteredIMG = filterColors(frame, isDay);

    Mat gray = applyGrayscale(filteredIMG);

    Mat gBlur = applyGaussianBlur(gray);

    Mat edges = applyCanny(gBlur);

    Mat maskedIMG = RegionOfInterest(edges);

    std::vector<Vec4i> linesP = houghLines(maskedIMG, frame.clone(), false);
    last_inference_time_ms_ =
        (cv::getTickCount() - start_tick) * 1000.0 / cv::getTickFrequency();
    return drawLanes(frame, linesP);
}