#include "FringePattern.hpp"
#include <iostream>
#include <cmath>

FringePattern::FringePattern(cv::Size resolution, int px_f, int steps)
    : width(resolution.width), height(resolution.height), steps(steps) {
    n_fringes = std::floor(static_cast<double>(width) / px_f);
    fr_images.resize(steps);
    
    for (int i = 0; i < steps; ++i) {
        fr_images[i] = cv::Mat::zeros(height, width, CV_8UC1);
    }
    // create_fringe_image();
}

void FringePattern::set_steps(int s){
    steps = s;
    fr_images.clear();
    fr_images.resize(steps);
    
    for (int i = 0; i < steps; ++i) {
        fr_images[i] = cv::Mat::zeros(height, width, CV_8UC1);
    }
}

void FringePattern::set_px_f(int pixel_f){
    px_f = pixel_f;
    n_fringes = std::floor(static_cast<double>(width) / px_f);

}

void FringePattern::show_fr_image() const {
    for (size_t i = 0; i < fr_images.size(); ++i) {
        cv::imshow("Image", fr_images[i]);
        cv::waitKey(0);
    }
    cv::destroyWindow("Image");
}

void FringePattern::print_image() const {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << static_cast<int>(fr_images[0].at<uchar>(i, j));
        }
        std::cout << "\n";
    }
    std::cout << "finished\n";
}

void FringePattern::create_fringe_image() {
    sin_values.clear();
    for (int n = 0; n < steps; ++n) {
        double phase_shift = n * 2.0 * CV_PI / steps;
        std::vector<double> current_sin(width);
        for (int x = 0; x < width; ++x) {
            current_sin[x] = std::sin(2.0 * CV_PI * n_fringes * x / width + phase_shift) + 1.0;
        }
        sin_values.push_back(current_sin);
    }

    // Preenchimento otimizado da matriz usando ponteiros
    for (int k = 0; k < steps; ++k) {
        for (int i = 0; i < height; ++i) {
            uchar* row_ptr = fr_images[k].ptr<uchar>(i);
            for (int j = 0; j < width; ++j) {
                row_ptr[j] = static_cast<uchar>(sin_values[k][j] * 255.0 / 2.0);
            }
        }
    }
}


std::vector<cv::Mat> FringePattern::get_fr_images(const std::string& color) const {
    std::vector<cv::Mat> colored_images(steps);
    cv::Mat zero_channel = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < steps; ++i) {
        std::vector<cv::Mat> canais(3, zero_channel);
        
        if (color == "blue") {
            canais[0] = fr_images[i];
        } else if (color == "green") {
            canais[1] = fr_images[i];
        } else if (color == "red") {
            canais[2] = fr_images[i];
        } else {
            return fr_images; // Retorna cinza por padrão
        }

        cv::merge(canais, colored_images[i]);
    }
    return colored_images;
}