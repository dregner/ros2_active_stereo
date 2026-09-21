#ifndef StereoFringeProcess_HPP
#define StereoFringeProcess_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <FringeProcess.hpp>
#include <opencv2/opencv.hpp>
#include <monitor_utils.hpp>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
#include <queue>

namespace ros2_active_stereo
{

class StereoFringeProcess : public rclcpp::Node {
public:
    StereoFringeProcess(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~StereoFringeProcess() override;

private:
    // ── Screen & Window Helpers ───────────────────────────────────────────
    bool get_screen_resolution(const std::string& monitor_name);
    void construct_window();
    void rebuild_patterns();

    // ── Acquisition Worker Thread ─────────────────────────────────────────
    void run_acquisition();
    void send_trigger();

    // ── Per-camera image queues (replaces ApproximateTimeSynchronizer) ────
    // Each callback pushes one cv::Mat; run_acquisition pops them.
    void left_image_cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void right_image_cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    cv::Mat pop_left(std::chrono::milliseconds timeout);
    cv::Mat pop_right(std::chrono::milliseconds timeout);

    // ── Other Callbacks ───────────────────────────────────────────────────
    void camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void process_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    void project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                    const std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    void save_img_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    void display_timer_cb();

    void publish_processed_images(const std::vector<cv::Mat>& images);

    // ── Fringe Engine ─────────────────────────────────────────────────────
    std::unique_ptr<FringeProcess> fringe_process_ptr_;

    int pixel_per_fringe_{128};
    int fringe_steps_{4};
    int settle_ms_{22};
    std::string color_{"blue"};

    cv::Size project_resolution_;
    std::string window_name_{"fringe"};
    std::pair<int, int> window_position_;

    std::vector<cv::Mat> all_imgs_;
    cv::Mat black_img_;

    std::atomic<bool> receive_camera_info_{false};
    std::atomic<bool> acquiring_{false};
    cv::Size camera_resolution_{2448, 2048};   // updated by camera_info_cb

    int manual_project_idx_{0};
    std::thread acquisition_thread_;

    // ── Per-camera image queues ────────────────────────────────────────────
    std::queue<cv::Mat> left_queue_;
    std::queue<cv::Mat> right_queue_;
    std::mutex          left_mtx_;
    std::mutex          right_mtx_;
    std::condition_variable left_cv_;
    std::condition_variable right_cv_;

    // ── ROS Infrastructure ────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     sub_left_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr     sub_right_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_left_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_right_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mod_left_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mod_right_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_left_debug_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_right_debug_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr scan_done_pub_;

    rclcpp::CallbackGroup::SharedPtr display_cb_group_;
    rclcpp::CallbackGroup::SharedPtr srv_cb_group_;
    rclcpp::CallbackGroup::SharedPtr cam_cb_group_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr change_image_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  process_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  save_imgs_service_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr   trigger_client_;

    rclcpp::TimerBase::SharedPtr display_timer_;
};

} // namespace ros2_active_stereo

#endif // StereoFringeProcess_HPP