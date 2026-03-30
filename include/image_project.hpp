#ifndef ImageProjectNode_HPP
#define ImageProjectNode_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include "std_msgs/msg/int8.hpp"
#include <opencv2/opencv.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <FringePattern.hpp>
#include <GrayCode.hpp>
#include <monitor_utils.hpp>


class ImageProjectNode : public rclcpp::Node {
public:
    ImageProjectNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ImageProjectNode() override;

private:
    void get_screen_resolution(const std::string& monitor_name);
    void project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         const std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    void construct_window();
    void project_image_timer_cb();

    std::unique_ptr<FringePattern> fringe_ptr_;
    std::unique_ptr<GrayCode> graycode_ptr_;

    int pixel_per_fringe;
    int fringe_steps;
    std::string color_;

    int n_proj_{0};
    cv::Size project_resolution_;
    bool project_imgs_{false};

    std::string window_name_{"fringe"};
    std::pair<int, int> window_position_;

    std::vector<cv::Mat> all_imgs_;
    cv::Mat black_img_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr change_image_service_;
    rclcpp::TimerBase::SharedPtr timer_;
};
#endif // ImageProjectNode_HPP