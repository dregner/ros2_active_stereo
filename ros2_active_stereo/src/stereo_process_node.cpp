#include <rclcpp/rclcpp.hpp>
#include "stereo_process.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoProcessNode>());
    rclcpp::shutdown();
    return 0;
}