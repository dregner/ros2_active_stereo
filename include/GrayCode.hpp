#ifndef GRAYCODE_HPP
#define GRAYCODE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class GrayCode {
public:
    GrayCode(cv::Size resolution = cv::Size(512, 512), int axis = 0, int px_f = 16);
    std::vector<cv::Mat> get_gc_images() const;
    std::vector<cv::Mat> get_color_gc_image(const std::string& color) const;
    void show_gc_image() const;
    void create_graycode_image();
    std::vector<std::string> list_to_graycode_binary(const std::vector<int>& int_list, int bit_length) const;
    std::vector<int> get_gc_order_v() const;

private:
    int width;
    int height;
    int px_f;
    int n_bits;
    std::vector<cv::Mat> gc_images;
};

#endif // GRAYCODE_HPP