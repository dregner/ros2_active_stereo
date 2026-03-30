#include <rclcpp/rclcpp.hpp>
#include "image_project.hpp" // Ajuste o nome se seu header for diferente

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProjectNode>());
    rclcpp::shutdown();
    return 0;
}