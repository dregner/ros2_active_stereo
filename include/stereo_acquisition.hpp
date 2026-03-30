#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <vector>
#include <utility>

class StereoAcquisitionNode : public rclcpp::Node
{
public:
    explicit StereoAcquisitionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    void stereo_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg);
    
    void request_next_projection(bool turn_on);
    void acquire_sb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                    const std::shared_ptr<std_srvs::srv::SetBool::Response> response);
    // --- ROS 2 INTERFACES ---
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_right_;

    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Service clients
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr projector_client_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr trigger_client_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr acquire_service_;

    // Image buffer
    using ImagePtr = sensor_msgs::msg::Image::ConstSharedPtr;
    std::vector<std::pair<ImagePtr, ImagePtr>> stereo_buffer_;

    int total_patterns_;  // Check number of pattern arrived
    bool acquisition_active_;
};
