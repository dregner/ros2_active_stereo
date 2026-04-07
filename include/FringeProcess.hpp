#ifndef FRINGEPROCESS_HPP
#define FRINGEPROCESS_HPP

#include "FringePattern.hpp"
#include "GrayCode.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <set>
class FringeProcess : public FringePattern, public GrayCode {
public:
    FringeProcess(cv::Size img_res = cv::Size(1920, 1080), 
                  cv::Size cam_res = cv::Size(1600, 1200), 
                  int px_f = 12, int steps = 12);

    void set_images(const cv::Mat& left, const cv::Mat& right, int counter);

    // Retorna um par de Mat (Modulation Map, Phi Image)
    std::pair<cv::Mat, cv::Mat> calculate_phi(const std::vector<cv::Mat>& images);

    cv::Mat calculate_qsi(const std::vector<cv::Mat>& graycode_images);

    cv::Mat remap_qsi_image(const cv::Mat& qsi_image, const std::vector<int>& real_qsi_order);

    void set_camera_resolution(cv::Size cam_res) {
        cam_width = cam_res.width;
        cam_height = cam_res.height;
    };

    int get_total_steps();
    // Retorna {abs_phi_l, abs_phi_r, mod_l, mod_r}
    std::vector<cv::Mat> calculate_abs_phi_images();

private:
    int n_bits;
    int cam_width, cam_height;
    int total_steps;

    
    // Armazenamento das capturas
    std::vector<cv::Mat> images_left;
    std::vector<cv::Mat> images_right;
};

#endif