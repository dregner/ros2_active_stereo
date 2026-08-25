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
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

/*
 * StereoCorrelProcess
 * -------------------
 * Manages the motor-scan + laser + camera trigger loop for spatial correlation.
 *
 * Image-loss fix
 * ~~~~~~~~~~~~~~
 * The BFS-U3-50S4M-C drops the first 1–2 frames after arming. Instead of
 * over-requesting images with a magic "+10" offset, we fire a configurable
 * number of "warmup" triggers first (default: 2) while service_request_ is
 * still false, then arm the save flag and fire exactly num_images_ triggers.
 * This gives a clean, deterministic acquisition count.
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

        // RAM directories for acquired images
        std::filesystem::create_directories("/tmp/rrp_stereo/left");
        std::filesystem::create_directories("/tmp/rrp_stereo/right");

        // Parameters
        this->declare_parameter<int>("num_images",      10);
        this->declare_parameter<int>("steps",           20);
        // Number of warmup triggers fired before arming the save flag.
        // The BFS camera typically drops 1-2 frames on first trigger after
        // hardware-trigger arming; 2 is a safe default.
        this->declare_parameter<int>("warmup_triggers", 2);

        num_images_      = this->get_parameter("num_images").as_int();
        steps_           = this->get_parameter("steps").as_int();
        warmup_triggers_ = this->get_parameter("warmup_triggers").as_int();

        // Publisher
        handshake_images_pub_ =
            this->create_publisher<std_msgs::msg::Int16>("handshake_images", 10);

        // Subscribers QoS
        auto qos = rclcpp::SensorDataQoS();
        qos.keep_last(15);
        rclcpp::SubscriptionOptions sub_options;

        left_sub_.subscribe(this, "left/image",  qos.get_rmw_qos_profile(), sub_options);
        right_sub_.subscribe(this, "right/image", qos.get_rmw_qos_profile(), sub_options);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                    SyncPolicy(15), left_sub_, right_sub_);
        sync_->registerCallback(
            std::bind(&StereoCorrelProcess::images_cb, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Services / clients
        service_request_ = false;
        perform_correl_  = false;

        cb_group_srv_    = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_client_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto srv_qos = rclcpp::ServicesQoS();
        srv_ = this->create_service<std_srvs::srv::SetBool>(
            "correlation_process",
            std::bind(&StereoCorrelProcess::get_images_srv, this,
                      std::placeholders::_1, std::placeholders::_2),
            srv_qos, cb_group_srv_);

        save_im_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "save_images_ssd",
            std::bind(&StereoCorrelProcess::save_images_ssd_srv, this,
                      std::placeholders::_1, std::placeholders::_2));

        gpio_client_ = this->create_client<std_srvs::srv::Trigger>(
            "trigger", srv_qos, cb_group_client_);

        laser_client_ = this->create_client<std_srvs::srv::SetBool>(
            "laser", srv_qos, cb_group_client_);

        motor_client_ = this->create_client<ros2_active_stereo_msgs::srv::MoveMotor>(
            "move_motor", srv_qos, cb_group_client_);
    }

private:

    using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    // ── Attributes ────────────────────────────────────────────────────────
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    uint8_t  count_;
    bool     service_request_;
    bool     perform_correl_;
    int      num_images_;
    int      steps_;
    int      warmup_triggers_;   // frames to discard before arming save

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

        service_request_ = false;
        count_ = 0;

        if (watchdog_timer_) {
            watchdog_timer_->cancel();
        }
        RCLCPP_INFO(this->get_logger(), "Handshake sent: %d images", count_to_send);
    }

    void watchdog_timeout_cb()
    {
        RCLCPP_WARN(this->get_logger(),
                    "Watchdog: timeout waiting for images. Saved %d / %d.",
                    count_, num_images_);
        send_handshake(count_);
    }

    // Helper to send a single trigger and wait synchronously.
    // Returns true on success.
    bool send_trigger_sync(std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
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
        if (!service_request_) return;

        try {
            cv::Mat left  = cv_bridge::toCvShare(left_msg,  "mono8")->image;
            cv::Mat right = cv_bridge::toCvShare(right_msg, "mono8")->image;

            if (left.empty() || right.empty()) {
                RCLCPP_WARN(this->get_logger(), "Empty frame received");
                return;
            }

            char left_fn[256], right_fn[256];
            snprintf(left_fn,  sizeof(left_fn),  "/tmp/rrp_stereo/left/L%02d.png",  count_ + 1);
            snprintf(right_fn, sizeof(right_fn), "/tmp/rrp_stereo/right/R%02d.png", count_ + 1);

            cv::imwrite(left_fn,  left);
            cv::imwrite(right_fn, right);

            count_++;
            RCLCPP_DEBUG(this->get_logger(), "Saved pair %d / %d", count_, num_images_);

            if (count_ == static_cast<uint8_t>(num_images_)) {
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

        const std::string ram_path  = "/dev/shm/stereo_active/";
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
        // Re-read parameters in case they changed since construction
        num_images_      = this->get_parameter("num_images").as_int();
        steps_           = this->get_parameter("steps").as_int();
        warmup_triggers_ = this->get_parameter("warmup_triggers").as_int();

        count_          = 0;
        service_request_ = false;   // will be armed after warmup
        perform_correl_  = request->data;

        const float angle_per_step = (steps_ / 2048.0f) * 360.0f;
        // Total motor steps = warmup + real acquisitions
        const int total_triggers = warmup_triggers_ + num_images_;

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

        // ── 2. Scan loop ──────────────────────────────────────────────────
        auto motor_req = std::make_shared<ros2_active_stereo_msgs::srv::MoveMotor::Request>();
        motor_req->angle = angle_per_step;

        for (int i = 0; i < total_triggers; i++) {

            // Move motor one step
            auto fut_motor = motor_client_->async_send_request(motor_req);
            if (fut_motor.wait_for(std::chrono::milliseconds(500)) != std::future_status::ready) {
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

            // Arm save flag after warmup triggers are done
            if (i == warmup_triggers_ - 1) {
                service_request_ = true;
                RCLCPP_INFO(this->get_logger(),
                            "Warmup done (%d triggers). Arming image save.", warmup_triggers_);
            }

            // Send camera trigger
            if (!send_trigger_sync()) {
                response->success = false;
                response->message = "Trigger failed at step " + std::to_string(i);
                return;
            }

            // Small delay — camera exposure time + DMA transfer margin
            rclcpp::sleep_for(std::chrono::milliseconds(10));
        }

        // ── 3. Turn off laser ─────────────────────────────────────────────
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
            if (!fut.get()->success) {
                RCLCPP_WARN(this->get_logger(), "Laser OFF returned failure (non-fatal).");
            }
            RCLCPP_INFO(this->get_logger(), "Laser OFF");
        }

        // ── 4. Return motor to home ───────────────────────────────────────
        {
            // Total displacement = num_steps_moved * angle_per_step (positive)
            // We moved total_triggers steps forward, so return is negative
            const float return_angle =
                -(static_cast<float>(total_triggers) * angle_per_step);
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
            RCLCPP_INFO(this->get_logger(), "Motor returned to home.");
        }

        // ── 5. Start watchdog — fire only if images haven't arrived yet ───
        if (service_request_) {
            watchdog_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(2500),
                std::bind(&StereoCorrelProcess::watchdog_timeout_cb, this),
                cb_group_srv_);
        }

        response->success = true;
        response->message = "Scan complete (" + std::to_string(total_triggers) +
                            " triggers, " + std::to_string(warmup_triggers_) +
                            " warmup, " + std::to_string(num_images_) + " saved)";
    }
};

} // namespace ros2_active_stereo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoCorrelProcess)