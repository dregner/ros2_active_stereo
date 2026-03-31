#ifndef ImageProjectNode_HPP
#define ImageProjectNode_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int8.hpp>
#include <opencv2/opencv.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <FringePattern.hpp>
#include <GrayCode.hpp>
#include <monitor_utils.hpp>


class ImageProjectNode : public rclcpp::Node {
public:
    ImageProjectNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ImageProjectNode() override;

private:
    bool get_screen_resolution(const std::string& monitor_name);
    void project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         const std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    void construct_window();
    void project_image_timer_cb();
    void trigger_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

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
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr n_imgs_pub_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr idx_proj_sub_;


    // For debug only
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_service_;


}; 
#endif // ImageProjectNode_HPP