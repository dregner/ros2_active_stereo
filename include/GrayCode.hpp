#ifndef GRAYCODE_HPP
#define GRAYCODE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class GrayCode {
public:
    GrayCode(cv::Size resolution = cv::Size(512, 512), int axis = 0, int px_f = 16);
    void show_gc_image() const;
    void create_graycode_image();
    std::vector<std::string> list_to_graycode_binary(const std::vector<int>& int_list, int bit_length) const;
    
    void set_axis(int a) { this->axis = a; };
    void set_px_f(int pixel_f) { this->px_f = pixel_f; };
    void set_resolution(cv::Size res) { this->width = res.width; this->height = res.height; };

    int get_n_bits() const { return n_bits; };
    int get_axis() const { return axis; };
    int get_px_f() const { return px_f; };

    std::vector<int> get_gc_order_v() const;
    std::vector<cv::Mat> get_gc_images(const std::string& color) const;

private:
    int width;
    int height;
    int px_f;
    int n_bits;
    int axis;
    std::vector<cv::Mat> gc_images;
};

#endif // GRAYCODE_HPP