#ifndef FRINGEPATTERN_HPP
#define FRINGEPATTERN_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class FringePattern {
public:
    FringePattern(cv::Size resolution = cv::Size(1024, 768), int px_f = 20, int steps = 4);

    void show_fr_image() const;
    void print_image() const;
    void create_fringe_image();
    std::vector<cv::Mat> get_fr_images(const std::string& color) const;
    
    int get_steps() const { return steps; };
    double get_n_fringes() const { return n_fringes; };
    
    void set_steps(int s);
    void set_px_f(int pixel_f);
    void set_resolution(cv::Size res) { this->width = res.width; this->height = res.height; };

private:
    int width;
    int height;
    double n_fringes;
    int steps;
    int px_f;
    std::vector<std::vector<double>> sin_values;
    std::vector<cv::Mat> fr_images;
};

#endif // FRINGEPATTERN_HPP