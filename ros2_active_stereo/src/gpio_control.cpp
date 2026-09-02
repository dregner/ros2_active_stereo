#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <gpiod.h>
#include <iostream>
#include <atomic>
#include "ros2_active_stereo_msgs/srv/move_motor.hpp"

class GpioControl : public rclcpp::Node
{
public:
    GpioControl() : Node("gpio_control_node"), keep_rotating_(false)
    {
        // Declare parameters
        this->declare_parameter<std::string>("stepping_mode", "full"); // full or half
        this->declare_parameter<int>("steps_per_revolution", 2048); // 4096 - half, 2048 - full
        this->declare_parameter<int>("delay", 3500); // Delay in microseconds


        // Get parameters
        this->get_parameter("steps_per_revolution", steps_per_revolution_);
        this->get_parameter("stepping_mode", stepping_mode_);
        this->get_parameter("delay", delay_);

        // Initialize GPIO lines
        chip0 = gpiod_chip_open(gpio_chip0_.c_str());
        if (!chip0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GPIO chip: %s", gpio_chip0_.c_str());
            throw std::runtime_error("Failed to open GPIO chip 0");
        }

        chip1 = gpiod_chip_open(gpio_chip1_.c_str());
        if (!chip1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GPIO chip: %s", gpio_chip1_.c_str());
            if (chip0) gpiod_chip_close(chip0);
            throw std::runtime_error("Failed to open GPIO chip 1");
        }

        for (int pin : gpio_pins_)
        {
            struct gpiod_chip *current_chip = chip1;
            if (pin == 85 || pin == 106) {
                current_chip = chip0;
            }

            struct gpiod_line *line = gpiod_chip_get_line(current_chip, pin);
            if (!line) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get GPIO line: %d", pin);
                if (chip0) gpiod_chip_close(chip0);
                if (chip1) gpiod_chip_close(chip1);
                throw std::runtime_error("Failed to get GPIO line");
            }
            if (pin == 85 || pin == 106){
                if(pin == 106){ 
                laser_line = line; 
                    int ret_la = gpiod_line_request_output(laser_line, "laser", 0);
                } // 106 = PQ.06 = MCLK05
                else{ 
                    trigger_line = line; 
                    int ret_tr = gpiod_line_request_output(trigger_line, "trigger", 0);
                } // 85 = PN.01 = GPIO27
            }
            else{ 
                gpio_lines_.push_back(line);
                int ret = gpiod_line_request_output(line, "stepper_motor", 0);
                if (ret < 0) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to request line as output: %d", pin);
                    if (chip0) gpiod_chip_close(chip0);
                    if (chip1) gpiod_chip_close(chip1);
                    throw std::runtime_error("Failed to request line as output");
                } 
            }
        }

        /* Subscribe to the topic
        subscription_ = this->create_subscription<std_msgs::msg::Float32>(
            "motor/angle", 10, std::bind(&GpioControl::move_motor, this, std::placeholders::_1));
        */

        motor_srv_ = this->create_service<ros2_active_stereo_msgs::srv::MoveMotor>(
            "move_motor", std::bind(&GpioControl::move_motor_cb, this, std::placeholders::_1, std::placeholders::_2));

        trigger_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "trigger", std::bind(&GpioControl::trigger_cb, this, std::placeholders::_1, std::placeholders::_2));

        laser_srv_ = this->create_service<std_srvs::srv::SetBool>(
            "laser", std::bind(&GpioControl::laser_cb, this, std::placeholders::_1, std::placeholders::_2));
    }

    ~GpioControl()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down GPIO control node");
        keep_rotating_ = false;
        if (rotation_thread_.joinable()) {
            rotation_thread_.join();
        }
        for (auto line : gpio_lines_) {
            gpiod_line_set_value(line, 0); // Set line to 0 before releasing
            gpiod_line_release(line);
        }
        if (laser_line) {
            gpiod_line_set_value(laser_line, 0); // Set laser line to 0 before releasing
            gpiod_line_release(laser_line);
        }
        if (trigger_line) {
            gpiod_line_set_value(trigger_line, 0); // Set trigger line to 0 before releasing
            gpiod_line_release(trigger_line);
        }
        if (chip0) gpiod_chip_close(chip0);
        if (chip1) gpiod_chip_close(chip1);
    }

private:
    void move_motor_cb(const ros2_active_stereo_msgs::srv::MoveMotor::Request::SharedPtr request,
                 ros2_active_stereo_msgs::srv::MoveMotor::Response::SharedPtr response){

        // Get parameters
        this->get_parameter("steps_per_revolution", steps_per_revolution_);
        this->get_parameter("stepping_mode", stepping_mode_);
        this->get_parameter("delay", delay_);
        
        std::vector<std::vector<int>> step_sequence;

        if (stepping_mode_ == "full") {
            step_sequence = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        } else {
            step_sequence = {
                {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 1, 0},
                {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 1}, {1, 0, 0, 1}};
        }

        float angle = request->angle;
        //RCLCPP_INFO(this->get_logger(), "Moving motor to angle: %f", angle);

        if (angle == 400.0 || angle == -400.0) {
            RCLCPP_INFO(this->get_logger(), "Continuous rotation mode");
            keep_rotating_ = false;
            if (rotation_thread_.joinable()) {
            rotation_thread_.join();
            }
            keep_rotating_ = true;
            bool clockwise = (angle == 400.0);
            if (!clockwise) {
            std::reverse(step_sequence.begin(), step_sequence.end());
            }
            rotation_thread_ = std::thread([this, step_sequence]() {
            while (keep_rotating_) {
                for (const auto& step : step_sequence) {
                for (size_t j = 0; j < gpio_lines_.size(); ++j) {
                    gpiod_line_set_value(gpio_lines_[j], step[j]);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(delay_));
                }
            }
            });
        } 
        if(angle == 500.0){
            if (rotation_thread_.joinable()) {
                rotation_thread_.join();
            }
            keep_rotating_ = true;
            std::vector<std::vector<int>> step_sequence_ccw = step_sequence;
            std::reverse(step_sequence_ccw.begin(), step_sequence_ccw.end());

            rotation_thread_ = std::thread([this, step_sequence, step_sequence_ccw](){
                int max_steps = static_cast<int>((110.0 / 360.0) * steps_per_revolution_);
            int step_counter = 0;
            bool clockwise = false;
            while (keep_rotating_) {
                // Choose direction
                const auto& seq = clockwise ? step_sequence : step_sequence_ccw;
                for (int i = 0; i < 10 && keep_rotating_; ++i) { // 10 steps per cycle
                    for (const auto& step : seq) {
                        for (size_t j = 0; j < gpio_lines_.size(); ++j) {
                            gpiod_line_set_value(gpio_lines_[j], step[j]);
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(delay_));
                    }
                    step_counter++;
                    if (step_counter >= max_steps) {
                        clockwise = !clockwise; // Switch direction
                        step_counter = 0;
                        break;
                    }
                }
            }
        });

        }
        else {
            keep_rotating_ = false;
            if (rotation_thread_.joinable()) {
                rotation_thread_.join();
            }

            int steps = static_cast<int>((angle / 360.0) * steps_per_revolution_);
            if (steps == 0) {
                stop_motor();
                response->success = true;
                RCLCPP_INFO(this->get_logger(), "Motor stopped (0 angle received).");
                return;
            }

            bool clockwise = (steps > 0);
            steps = std::abs(steps);

            if (!clockwise) {
                std::reverse(step_sequence.begin(), step_sequence.end());
            }

            for (int i = 0; i < steps; ++i) {
                const auto& step = step_sequence[i % step_sequence.size()];
                for (size_t j = 0; j < gpio_lines_.size(); ++j) {
                    gpiod_line_set_value(gpio_lines_[j], step[j]);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(delay_));
            }
            // RCLCPP_INFO(this->get_logger(), "Motor movement complete.");
        }
        
        response->success = true;
    }

    void stop_motor(){
        for (auto line : gpio_lines_) {
            gpiod_line_set_value(line, 0);
        }
    }

    void trigger_cb(const std_srvs::srv::Trigger::Request::SharedPtr request,
                    const std_srvs::srv::Trigger::Response::SharedPtr response){
        // RCLCPP_INFO(this->get_logger(), "Trigger service called! Sending pulse...");
        gpiod_line_set_value(trigger_line, 1);
        rclcpp::sleep_for(std::chrono::microseconds(500));
        gpiod_line_set_value(trigger_line, 0);
        // RCLCPP_INFO(this->get_logger(), "Trigger pulse sent.");
        response->success = true;
    }

    void laser_cb(const std_srvs::srv::SetBool::Request::SharedPtr request,
                 const std_srvs::srv::SetBool::Response::SharedPtr response){
        if (request->data){
            gpiod_line_set_value(laser_line, 1);
            RCLCPP_INFO(this->get_logger(), "Laser ON: %d", gpiod_line_get_value(laser_line));
            response->message = "Laser ON";
            response->success = true;
        }
        else if(!request->data){
            gpiod_line_set_value(laser_line, 0);
            RCLCPP_INFO(this->get_logger(), "Laser OFF: %d", gpiod_line_get_value(laser_line));
            response->message = "Laser OFF";
            response->success = true;
        }
        else
        {
            response->success = false;
        }
    }

    std::vector<int> gpio_pins_ = {1, 0, 8, 2, 85, 106}; // Replace with your actual GPIO pin numbers
    std::string gpio_chip0_ = "/dev/gpiochip0";
    std::string gpio_chip1_ = "/dev/gpiochip1";
    std::string stepping_mode_;
    std::vector<std::vector<int>> step_sequence_;
    struct gpiod_chip *chip0 = nullptr;
    struct gpiod_chip *chip1 = nullptr;
    std::vector<struct gpiod_line *> gpio_lines_;
    struct gpiod_line *laser_line;
    struct gpiod_line *trigger_line;
    int delay_;
    int steps_per_revolution_;
    
    //rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr laser_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_srv_;
    rclcpp::Service<ros2_active_stereo_msgs::srv::MoveMotor>::SharedPtr motor_srv_;

    std::atomic<bool> keep_rotating_;
    std::thread rotation_thread_;

};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GpioControl>());
    rclcpp::shutdown();
    return 0;
}
