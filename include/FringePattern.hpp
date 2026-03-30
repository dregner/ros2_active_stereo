#ifndef FRINGEPATTERN_HPP
#define FRINGEPATTERN_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class FringePattern {
public:
    FringePattern(cv::Size resolution = cv::Size(1024, 768), int px_f = 20, int steps = 4);

    int get_steps() const;
    void show_fr_image() const;
    void print_image() const;
    void create_fringe_image();
    std::vector<cv::Mat> get_fr_images() const;
    std::vector<cv::Mat> get_color_fr_image(const std::string& color) const;
    

private:
    int width;
    int height;
    double n_fringes;
    int steps;
    std::vector<std::vector<double>> sin_values;
    std::vector<cv::Mat> fr_images;
};

#endif // FRINGEPATTERN_HPP