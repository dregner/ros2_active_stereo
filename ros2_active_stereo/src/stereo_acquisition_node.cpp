#include <rclcpp/rclcpp.hpp>
#include "stereo_acquisition.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoAcquisitionNode>());
    rclcpp::shutdown();
    return 0;
}