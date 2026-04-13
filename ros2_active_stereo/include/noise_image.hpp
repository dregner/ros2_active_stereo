#ifndef IMAGE_DISPLAY_NODE_H
#define IMAGE_DISPLAY_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include <random>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <noise/noise.h>
#include <noise/noiseutils.h>

#include <monitor_utils.hpp>

class ImageDisplayNode : public rclcpp::Node {
public:
    ImageDisplayNode();

private:
    void get_screen_resolution(const std::string& monitor_name);
    void construct_window();
    cv::Mat generate_noise_image(int seed);
    void change_image_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         const std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    void timer_image_cb();
    int generate_random_seed();

    double frequency_;
    double persistence_;
    double lacunarity_;
    double octave_;

    std::string monitor_name_;
    std::string window_name_;
    std::pair<int, int> window_position_;
    cv::Mat image_;
    int image_width_;
    int image_height_;
    int img_counter_;
    
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr change_image_service_;
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif // IMAGE_DISPLAY_NODE_H