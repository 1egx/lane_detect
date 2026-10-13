#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>

using namespace cv;


template <typename T, typename X>
X multiplyAndSum(std::vector<T> A, std::vector<T> B) {
    X sum;
    std::vector<T> temp;
    for (int i = 0; i < A.size(); i++)
    {
        temp.push_back(A[i]*B[i]);
    }
    sum = std::accumulate(temp.begin(), temp.end(), 0.0);

    return sum;
}

template <typename T, typename X>
std::vector< X > estimateCoefficients(std::vector<T> A, std::vector<T> B)
{
    // Sample size
    int N = A.size();

    // Calculate mean of X and Y
    X meanA= std::accumulate(A.begin(), A.end(), 0.0) / A.size();
    X meanB = std::accumulate(B.begin(), B.end(), 0.0) / B.size();

    X SSxy = multiplyAndSum<T, T>(A, B) - (N * meanA * meanB);
    X SSxx = multiplyAndSum<T, T>(A, A) - (N * meanA * meanA);

    X slopeB1 = SSxy / SSxx;
    X interceptB0 = meanB - (slopeB1 * meanA);

    std::vector< X > coef;
    coef.push_back(slopeB1);
    coef.push_back(interceptB0);
    return coef;
}

class Tradition {
private:
    Mat applyGrayscale(Mat source);

    /**
    * @brief Apply Gaussian blur to image.
    * 
    * @param source image that needs to be blurred
    * @return blurred image
    */
    Mat applyGaussianBlur(Mat source);

    /**
    * @brief Detect edges of image by applying canny edge detection.
    * 
    * @param source image of which the edges needs to be detected
    * @return image with detected edges
    */
    Mat applyCanny(Mat source);

    /**
    * @brief Filter source image so that only the white and yellow pixels remain.
    * A gray filter will also be added if the source image is classified as taken during the night.
    * One assumption for lane detection here is that lanes are either white or yellow.
    * 
    * @param source source image
    * @param isDayTime true if image is taken during the day, false if at night
    * @see isDayTime
    * @return Mat filtered image
    */
    Mat filterColors(Mat source, bool isDayTime);

    /**
    * @brief Apply an image mask. 
    * Only keep the part of the image defined by the
    * polygon formed from four points. The rest of the image is set to black.
    * 
    * @param source image on which to apply the mask
    * @return Mat image with mask
    */
    Mat RegionOfInterest(Mat source);

    /**
    * @brief Creates mask and blends it with source image so that the lanes
    * are drawn on the source image.
    * 
    * @param source source image
    * @param lines vector < vec4i > holding the lines
    * @return Mat image with lines drawn on it
    */
    Mat drawLanes(Mat source, std::vector<Vec4i> lines);

    /**
    * @brief Returns a vector with the detected hough lines.
    * 
    * @param canny image resulted from a canny transform
    * @param source image on which hough lines are drawn
    * @param drawHough draw detected lines on source image if true. 
    * It will also show the image with de lines drawn on it, which is why
    * it is not recommended to pass in true when working with a video. 
    * @see applyCanny
    * @return vector<Vec4i> contains hough lines.
    */
    std::vector<Vec4i> houghLines(Mat canny, Mat source, bool drawHough);

    /**
    * @brief Determine whether a picture is taken during day or night time.
    * Returns true when the image is classified as daytime. 
    * Note: this is based on the mean pixel value of an image and might not
    * always lead to accurate results.
    * 
    * @param source image
    * @return true 
    * @return false 
    */
    bool isDayTime(Mat source);

    
    double last_inference_time_ms_ = 0.0;
public:
    Mat Infer(Mat frame);
    double GetLastInferenceTimeMs() {return last_inference_time_ms_;}
};