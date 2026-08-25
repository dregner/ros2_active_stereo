#ifndef StereoFringeProcess_HPP
#define StereoFringeProcess_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <FringeProcess.hpp>
#include <opencv2/opencv.hpp>
#include <monitor_utils.hpp>
#include <chrono>
#include <atomic>
#include <mutex>

namespace ros2_active_stereo
{
class StereoFringeProcess : public rclcpp::Node {
public:
    StereoFringeProcess(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~StereoFringeProcess() override;

private:
    // ── Scan state machine ─────────────────────────────────────────────────
    enum class ScanState {
        IDLE,               // waiting for service call; projects black
        SETTLING,           // pattern projected; one-shot timer counting down
        WAITING_FOR_FRAME,  // trigger sent; waiting for stereo_callback
        PROCESSING          // last frame received; computing phase maps
    };

    ScanState scan_state_{ScanState::IDLE};
    std::mutex scan_mtx_;       // protects scan_state_ and scan_index_

    // Index of the pattern currently projected (0 = black warm-up)
    int scan_index_{0};

    // Parameters cached at start of each scan (avoid per-callback get_parameter)
    int cached_settle_ms_{22};  // ≈ 1 projector frame at 60 Hz + 5 ms margin
    bool cached_debug_{false};
    std::string cached_color_{"blue"};

    // ── Helpers ───────────────────────────────────────────────────────────
    bool get_screen_resolution(const std::string& monitor_name);
    void construct_window();

    // Start projecting pattern at scan_index_, arm one-shot settling timer
    void advance_scan_step();
    // Called by settling_timer_: send trigger, → WAITING_FOR_FRAME
    void settling_done_cb();

    void send_trigger();

    // ── Callbacks ─────────────────────────────────────────────────────────
    void camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                         const sensor_msgs::msg::Image::ConstSharedPtr& right_msg);

    // Service: start scan (returns immediately; result on scan_done_pub_)
    void process_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // Service: manual projection control (for testing/alignment)
    void project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                    const std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    void save_img_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         const std::shared_ptr<std_srvs::srv::Trigger::Response> response);

    // Idle / UI timer: keeps the OpenCV window alive and shows current image
    void display_timer_cb();

    // Publishing
    void publish_processed_images(const std::vector<cv::Mat>& images);

    // ── Fringe engine ─────────────────────────────────────────────────────
    std::unique_ptr<FringeProcess> fringe_process_ptr_;

    // Parameter variables (cached on construct; re-read only when changed)
    int pixel_per_fringe{128};
    int fringe_steps{4};
    double timer_hz_{16.0};   // display timer interval in ms (≤ projector period)
    std::string color_{"blue"};

    cv::Size project_resolution_;
    std::string window_name_{"fringe"};
    std::pair<int, int> window_position_;

    std::vector<cv::Mat> all_imgs_;
    cv::Mat black_img_;

    std::atomic<bool> receive_camera_info_{false};

    // ── ROS infrastructure ────────────────────────────────────────────────
    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_left_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_right_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Phase map publishers (64FC1)
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_left_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_right_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mod_left_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mod_right_;

    // Debug publishers (normalised 8-bit)
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_left_debug_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_abs_right_debug_;

    // Scan status publisher ("scan_started" / "scan_complete" / "scan_error")
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr scan_done_pub_;

    // Callback groups
    rclcpp::CallbackGroup::SharedPtr display_cb_group_;  // display timer
    rclcpp::CallbackGroup::SharedPtr srv_cb_group_;      // services + trigger client
    rclcpp::CallbackGroup::SharedPtr stereo_cb_group_;   // stereo message filter

    // Services / clients
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr change_image_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  process_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  save_imgs_service_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr   trigger_client_;

    // Display timer (keeps OpenCV window alive; does NOT send triggers)
    rclcpp::TimerBase::SharedPtr display_timer_;

    // One-shot settling timer (created fresh for every pattern)
    rclcpp::TimerBase::SharedPtr settling_timer_;
};
}
#endif // StereoFringeProcess_HPP