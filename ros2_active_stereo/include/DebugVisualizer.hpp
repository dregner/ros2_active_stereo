#ifndef DEBUG_VISUALIZER_HPP
#define DEBUG_VISUALIZER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class DebugVisualizer {
private:
    static inline cv::Mat safeRender1D(cv::Mat data, std::string title) {
        cv::Mat plot(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        
        if (data.empty() || data.cols < 2 || data.rows == 0) {
            cv::putText(plot, "DADOS INSUFICIENTES", cv::Point(200, 300), 0, 1, cv::Scalar(0,0,255), 2);
            return plot;
        }

        // Converte para double (CV_64F) garantindo que a leitura nunca puxe "lixo"
        cv::Mat data_double;
        data.convertTo(data_double, CV_64F);

        double min, max;
        cv::minMaxLoc(data_double, &min, &max);
        double range = (std::abs(max - min) < 1e-7) ? 1.0 : (max - min);

        std::vector<cv::Point> pts;
        int divisor_x = (data_double.cols > 1) ? (data_double.cols - 1) : 1;

        for (int i = 0; i < data_double.cols; i++) {
            double val = data_double.at<double>(0, i);
            int x = 60 + (i * (800 - 120) / divisor_x);
            int y = 540 - static_cast<int>((val - min) / range * 480.0);
            pts.push_back(cv::Point(x, y));
        }
        
        cv::polylines(plot, pts, false, cv::Scalar(50, 50, 50), 2);
        cv::putText(plot, title, cv::Point(60, 40), 0, 0.7, cv::Scalar(0,0,0), 2);
        return plot;
    }

    static inline cv::Mat safeRender2D(cv::Mat img, std::string title, int colormap) {
        cv::Mat canvas(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        if (img.empty() || img.rows == 0 || img.cols == 0) {
            cv::putText(canvas, "MATRIZ VAZIA", cv::Point(250, 300), 0, 1, cv::Scalar(0,0,255), 2);
            return canvas;
        }

        cv::Mat norm, colored;
        cv::normalize(img, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(norm, colored, colormap);
        
        cv::Mat resized;
        cv::resize(colored, resized, cv::Size(740, 480));
        resized.copyTo(canvas(cv::Rect(30, 90, 740, 480)));
        cv::putText(canvas, title, cv::Point(30, 50), 0, 0.8, cv::Scalar(0,0,0), 2);
        return canvas;
    }

public:
    static inline void saveDebugMosaic(std::string filename, 
                                       cv::Mat phi_l, cv::Mat phi_r,
                                       cv::Mat mod_l, cv::Mat mod_r) {
        
        cv::Mat row_l = (phi_l.rows > 0) ? phi_l.row(phi_l.rows / 2) : cv::Mat();
        cv::Mat row_r = (phi_r.rows > 0) ? phi_r.row(phi_r.rows / 2) : cv::Mat();

        cv::Mat p1 = safeRender1D(row_l, "1D Phase Left");
        cv::Mat p2 = safeRender1D(row_r, "1D Phase Right");
        cv::Mat p3 = safeRender2D(phi_l, "2D Phase Left", cv::COLORMAP_BONE);
        cv::Mat p4 = safeRender2D(phi_r, "2D Phase Right", cv::COLORMAP_BONE);
        cv::Mat p5 = safeRender2D(mod_l, "Modulation Left", cv::COLORMAP_JET);
        cv::Mat p6 = safeRender2D(mod_r, "Modulation Right", cv::COLORMAP_JET);

        cv::Mat r1, r2, r3, mosaic;
        cv::hconcat(std::vector<cv::Mat>{p1, p2}, r1);
        cv::hconcat(std::vector<cv::Mat>{p3, p4}, r2);
        cv::hconcat(std::vector<cv::Mat>{p5, p6}, r3);
        
        std::vector<cv::Mat> rows = {r1, r2, r3};
        cv::vconcat(rows, mosaic);

        if (!mosaic.empty()) {
            cv::imwrite(filename, mosaic);
            std::cout << ">>> Mosaico de Debug exportado para: " << filename << std::endl;
        }
    }
};

#endif // DEBUG_VISUALIZER_HPP