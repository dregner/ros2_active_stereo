#ifndef StereoProcessNode_HPP
#define StereoProcessNode_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <FringeProcess.hpp>
#include <opencv2/opencv.hpp>
#include <monitor_utils.hpp>
#include <chrono>


class StereoProcessNode : public rclcpp::Node {
public:
    StereoProcessNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~StereoProcessNode() override;

private:
    bool get_screen_resolution(const std::string& monitor_name);

    void camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    
    void project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                    const std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    void send_trigger();
    void _trigger_callback(rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future);
    void stereo_callback( const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                          const sensor_msgs::msg::Image::ConstSharedPtr& right_msg);
    
    void process_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void construct_window();
    void project_image_timer_cb();


    std::unique_ptr<FringeProcess> fringe_process_ptr_;

    // Parameters variables
    int pixel_per_fringe;
    int fringe_steps;
    double timer_hz_;

    std::string color_;

    int n_proj_{0};
    int trigger_timer_{0};
    cv::Size project_resolution_;

    bool project_imgs_{false};
    bool process_{false};
    bool receive_camera_info_{false};

    std::string window_name_{"fringe"};
    std::pair<int, int> window_position_;

    std::vector<cv::Mat> all_imgs_;
    cv::Mat black_img_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_right_;

    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
    rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
    
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr change_image_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr process_service_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_client_;
    rclcpp::TimerBase::SharedPtr timer_;




}; 
#endif // StereoProcessNode_HPP