#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int16.hpp>
#include "ros2_active_stereo_msgs/srv/move_motor.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

/*
 * StereoCorrelProcess
 * -------------------
 * Manages the motor-scan + laser + camera trigger loop for spatial correlation.
 *
 * Deterministic Acquisition:
 * - Uses ApproximateTime synchronization (queue 50, 50ms tolerance) to prevent
 *   frame loss from USB camera clock jitter.
 * - Fires warmup triggers while the motor is stationary to wake up the camera sensor
 *   without moving the baseline.
 * - Arms image capture flag and executes exact num_images motor steps and triggers.
 * - Saves images directly to RAM (/tmp/rrp_stereo/) and sends handshake upon completion.
 */

namespace ros2_active_stereo
{
class StereoCorrelProcess : public rclcpp::Node {

public:
    explicit StereoCorrelProcess(const rclcpp::NodeOptions & options)
    : Node("inverse_correlation_node", options)
    {
        RCLCPP_INFO(this->get_logger(), "StereoCorrelProcess started");

        count_ = 0;
        service_request_ = false;
        perform_correl_  = false;

        // RAM directories for acquired images
        std::filesystem::create_directories("/tmp/rrp_stereo/left");
        std::filesystem::create_directories("/tmp/rrp_stereo/right");

        // Parameters
        this->declare_parameter<int>("num_images",      10);
        this->declare_parameter<int>("steps",           20);
        this->declare_parameter<int>("warmup_triggers", 2);

        num_images_      = this->get_parameter("num_images").as_int();
        steps_           = this->get_parameter("steps").as_int();
        warmup_triggers_ = this->get_parameter("warmup_triggers").as_int();

        // Publisher
        handshake_images_pub_ =
            this->create_publisher<std_msgs::msg::Int16>("handshake_images", 10);

        // Subscribers QoS with ample queue to prevent drops
        auto qos = rclcpp::SensorDataQoS();
        qos.keep_last(50);
        rclcpp::SubscriptionOptions sub_options;

        left_sub_.subscribe(this, "left/image",  qos.get_rmw_qos_profile(), sub_options);
        right_sub_.subscribe(this, "right/image", qos.get_rmw_qos_profile(), sub_options);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                    SyncPolicy(50), left_sub_, right_sub_);
        sync_->registerCallback(
            std::bind(&StereoCorrelProcess::images_cb, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Callback groups
        cb_group_srv_    = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_client_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        srv_ = this->create_service<std_srvs::srv::SetBool>(
            "correlation_process",
            std::bind(&StereoCorrelProcess::get_images_srv, this,
                      std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(), cb_group_srv_);

        save_im_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "save_images_ssd",
            std::bind(&StereoCorrelProcess::save_images_ssd_srv, this,
                      std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS());

        gpio_client_ = this->create_client<std_srvs::srv::Trigger>(
            "trigger", rclcpp::ServicesQoS(), cb_group_client_);

        laser_client_ = this->create_client<std_srvs::srv::SetBool>(
            "laser", rclcpp::ServicesQoS(), cb_group_client_);

        motor_client_ = this->create_client<ros2_active_stereo_msgs::srv::MoveMotor>(
            "move_motor", rclcpp::ServicesQoS(), cb_group_client_);
    }

private:

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    // ── Attributes ────────────────────────────────────────────────────────
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    std::mutex state_mtx_;
    uint8_t  count_{0};
    bool     service_request_{false};
    bool     perform_correl_{false};
    int      num_images_{10};
    int      steps_{20};
    int      warmup_triggers_{2};

    rclcpp::CallbackGroup::SharedPtr cb_group_srv_;
    rclcpp::CallbackGroup::SharedPtr cb_group_client_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr  srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr  save_im_srv_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr   gpio_client_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr   laser_client_;
    rclcpp::Client<ros2_active_stereo_msgs::srv::MoveMotor>::SharedPtr motor_client_;

    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr handshake_images_pub_;

    // ── Helpers ───────────────────────────────────────────────────────────

    void send_handshake(int count_to_send)
    {
        auto msg = std_msgs::msg::Int16();
        msg.data = perform_correl_ ? count_to_send : -count_to_send;
        handshake_images_pub_->publish(msg);

        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            service_request_ = false;
            count_ = 0;
        }

        if (watchdog_timer_) {
            watchdog_timer_->cancel();
        }
        RCLCPP_INFO(this->get_logger(), "Handshake sent: %d images", count_to_send);
    }

    void watchdog_timeout_cb()
    {
        int current_count = 0;
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            current_count = count_;
        }
        RCLCPP_WARN(this->get_logger(),
                    "Watchdog timeout: saved %d / %d images.",
                    current_count, num_images_);
        send_handshake(current_count);
    }

    bool send_trigger_sync(std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
        if (!gpio_client_->service_is_ready()) {
            RCLCPP_ERROR(this->get_logger(), "Trigger service not ready!");
            return false;
        }
        auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
        auto fut = gpio_client_->async_send_request(req);
        if (fut.wait_for(timeout) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Trigger service timed out!");
            return false;
        }
        if (!fut.get()->success) {
            RCLCPP_ERROR(this->get_logger(), "Trigger service returned failure!");
            return false;
        }
        return true;
    }

    // ── Image callback ─────────────────────────────────────────────────────
    void images_cb(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                   const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        std::lock_guard<std::mutex> lock(state_mtx_);
        if (!service_request_) return;

        try {
            cv::Mat left  = cv_bridge::toCvShare(left_msg,  "mono8")->image;
            cv::Mat right = cv_bridge::toCvShare(right_msg, "mono8")->image;

            if (left.empty() || right.empty()) {
                RCLCPP_WARN(this->get_logger(), "Empty frame received in images_cb");
                return;
            }

            char left_fn[256], right_fn[256];
            snprintf(left_fn,  sizeof(left_fn),  "/tmp/rrp_stereo/left/L%02d.png",  count_ + 1);
            snprintf(right_fn, sizeof(right_fn), "/tmp/rrp_stereo/right/R%02d.png", count_ + 1);

            cv::imwrite(left_fn,  left);
            cv::imwrite(right_fn, right);

            count_++;
            RCLCPP_INFO(this->get_logger(), "Saved pair %d / %d", count_, num_images_);

            if (count_ >= static_cast<uint8_t>(num_images_)) {
                RCLCPP_INFO(this->get_logger(),
                            "All %d image pairs saved successfully.", count_);
                send_handshake(count_);
            }

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    // ── save_images_ssd_srv ────────────────────────────────────────────────
    void save_images_ssd_srv(
        const std::shared_ptr<std_srvs::srv::Trigger::Request>  /*request*/,
        const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        const char* home = std::getenv("HOME");
        if (!home) {
            RCLCPP_ERROR(this->get_logger(), "HOME environment variable not set.");
            response->success = false;
            response->message = "HOME not found";
            return;
        }

        const std::string ram_path  = "/tmp/rrp_stereo/";
        const std::string dest_path = std::string(home) + "/Pictures/stereo_active_backup";
        RCLCPP_INFO(this->get_logger(), "Saving images to %s …", dest_path.c_str());

        try {
            std::filesystem::create_directories(dest_path);
            std::filesystem::copy(ram_path, dest_path,
                std::filesystem::copy_options::recursive |
                std::filesystem::copy_options::overwrite_existing);
            response->success = true;
            response->message = "Images saved";
            RCLCPP_INFO(this->get_logger(), "Images saved successfully.");
        } catch (const std::filesystem::filesystem_error& e) {
            RCLCPP_ERROR(this->get_logger(), "Copy error: %s", e.what());
            response->success = false;
            response->message = e.what();
        }
    }

    // ── get_images_srv ─────────────────────────────────────────────────────
    void get_images_srv(
        const std::shared_ptr<std_srvs::srv::SetBool::Request>  request,
        const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        // Re-read parameters
        num_images_      = this->get_parameter("num_images").as_int();
        steps_           = this->get_parameter("steps").as_int();
        warmup_triggers_ = this->get_parameter("warmup_triggers").as_int();

        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            count_           = 0;
            service_request_ = false;
            perform_correl_  = request->data;
        }

        const float angle_per_step = (steps_ / 2048.0f) * 360.0f;

        // ── 1. Turn on laser ──────────────────────────────────────────────
        {
            auto laser_req = std::make_shared<std_srvs::srv::SetBool::Request>();
            laser_req->data = true;
            auto fut = laser_client_->async_send_request(laser_req);
            if (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                RCLCPP_ERROR(this->get_logger(), "Laser service timed out!");
                response->success = false;
                response->message = "Laser timeout";
                return;
            }
            if (!fut.get()->success) {
                RCLCPP_ERROR(this->get_logger(), "Laser service failed!");
                response->success = false;
                response->message = "Laser failed";
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Laser ON");
        }

        // ── 2. Warmup triggers (stationary, flushes initial sensor dropped frames)
        if (warmup_triggers_ > 0) {
            RCLCPP_INFO(this->get_logger(), "Firing %d warmup triggers...", warmup_triggers_);
            for (int w = 0; w < warmup_triggers_; ++w) {
                send_trigger_sync();
                rclcpp::sleep_for(std::chrono::milliseconds(40));
            }
            // Allow any pending warmup frames to clear
            rclcpp::sleep_for(std::chrono::milliseconds(100));
        }

        // ── 3. Arm save flag ──────────────────────────────────────────────
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            count_ = 0;
            service_request_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "Acquisition armed for %d images (angle step: %.2f°)",
                    num_images_, angle_per_step);

        // ── 4. Main scan loop ─────────────────────────────────────────────
        auto motor_req = std::make_shared<ros2_active_stereo_msgs::srv::MoveMotor::Request>();
        motor_req->angle = angle_per_step;

        for (int i = 0; i < num_images_; i++) {

            // Move motor one step
            auto fut_motor = motor_client_->async_send_request(motor_req);
            if (fut_motor.wait_for(std::chrono::milliseconds(1000)) != std::future_status::ready) {
                RCLCPP_ERROR(this->get_logger(), "Motor service timed out at step %d!", i);
                response->success = false;
                response->message = "Motor timeout";
                return;
            }
            if (!fut_motor.get()->success) {
                RCLCPP_ERROR(this->get_logger(), "Motor step %d failed!", i);
                response->success = false;
                response->message = "Motor failed";
                return;
            }

            // Mechanical stabilization delay
            rclcpp::sleep_for(std::chrono::milliseconds(20));

            // Send camera trigger
            if (!send_trigger_sync()) {
                response->success = false;
                response->message = "Trigger failed at step " + std::to_string(i);
                return;
            }

            // Camera exposure + USB transfer margin
            rclcpp::sleep_for(std::chrono::milliseconds(30));
        }

        // ── 5. Turn off laser ─────────────────────────────────────────────
        {
            auto laser_req = std::make_shared<std_srvs::srv::SetBool::Request>();
            laser_req->data = false;
            auto fut = laser_client_->async_send_request(laser_req);
            if (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
                RCLCPP_ERROR(this->get_logger(), "Laser OFF timed out!");
                response->success = false;
                response->message = "Laser OFF timeout";
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Laser OFF");
        }

        // ── 6. Return motor to starting position ──────────────────────────
        {
            const float return_angle = -(static_cast<float>(num_images_) * angle_per_step);
            auto ret_req = std::make_shared<ros2_active_stereo_msgs::srv::MoveMotor::Request>();
            ret_req->angle = return_angle;
            RCLCPP_INFO(this->get_logger(), "Returning motor %.2f°", return_angle);

            auto fut_ret = motor_client_->async_send_request(ret_req);
            if (fut_ret.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
                RCLCPP_ERROR(this->get_logger(), "Motor return timed out!");
                response->success = false;
                response->message = "Motor return timeout";
                return;
            }
            if (!fut_ret.get()->success) {
                RCLCPP_ERROR(this->get_logger(), "Motor return failed!");
                response->success = false;
                response->message = "Motor return failed";
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Motor returned to home position.");
        }

        // ── 7. Start watchdog for final image collection ──────────────────
        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            if (service_request_) {
                watchdog_timer_ = this->create_wall_timer(
                    std::chrono::milliseconds(3000),
                    std::bind(&StereoCorrelProcess::watchdog_timeout_cb, this),
                    cb_group_srv_);
            }
        }

        response->success = true;
        response->message = "Scan complete (" + std::to_string(num_images_) + " images requested)";
    }
};

} // namespace ros2_active_stereo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoCorrelProcess)