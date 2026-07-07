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
tarefas realizadas por inv_correlation_node.py que devem ser realizadas agora por esse cpp:
    callbacks das imagens, subscribers e publishers, trigger/lasers services
    get_images_srv
*/

namespace ros2_active_stereo
{
class StereoCorrelProcess : public rclcpp::Node{

public:
    explicit StereoCorrelProcess(const rclcpp::NodeOptions & options) : Node("inverse_correlation_node", options){
        RCLCPP_INFO(this->get_logger(), "StereoCorrelProcess.cpp has been started");
        
        count_= 0;

        // pastas na RAM onde ficam salvas as imagens
        std::filesystem::create_directories("/tmp/rrp_stereo/left");
        std::filesystem::create_directories("/tmp/rrp_stereo/right");

        // parametros
        this->declare_parameter<int>("num_images",10);
        this->declare_parameter<int>("steps", 20);
        num_images_ = this->get_parameter("num_images").as_int() +3; // +2 porque a(s) primeira(s) imagem sempre e'/sao perdida(s)
        steps_ = this->get_parameter("steps").as_int();

        //publisher
        handshake_images_pub_ = this->create_publisher<std_msgs::msg::Int16>("handshake_images", 10);

        //Subscribers Quality of Service
        auto qos = rclcpp::SensorDataQoS();
        qos.keep_last(15);
        rclcpp::SubscriptionOptions sub_options;    

        // subscribers
        left_sub_.subscribe(this, "left/image", qos.get_rmw_qos_profile(), sub_options);
        right_sub_.subscribe(this, "right/image", qos.get_rmw_qos_profile(), sub_options);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(15), left_sub_, right_sub_);
        sync_->registerCallback(std::bind(&StereoCorrelProcess::images_cb, this, std::placeholders::_1, std::placeholders::_2));

        // services
        service_request_= false;
        perform_correl_ = false;
        cb_group_srv_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        //client
        cb_group_client_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        srv_ = this->create_service<std_srvs::srv::SetBool>(
            "correlation_process", 
            std::bind(&StereoCorrelProcess::get_images_srv, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            cb_group_srv_
        );
        save_im_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "save_images_ssd",
            std::bind(&StereoCorrelProcess::save_images_ssd_srv, this, std::placeholders::_1, std::placeholders::_2)
        );

        gpio_client_ = this->create_client<std_srvs::srv::Trigger>(
            "trigger",
            rmw_qos_profile_services_default,
            cb_group_client_
        );

        laser_client_ = this->create_client<std_srvs::srv::SetBool>(
            "laser",
            rmw_qos_profile_services_default,
            cb_group_client_
        );
        motor_client_ = this->create_client<ros2_active_stereo_msgs::srv::MoveMotor>(
            "move_motor", 
            rmw_qos_profile_services_default,
            cb_group_client_
        );
    }

private:

    using SyncPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    //atributos
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    uint8_t count_;
    bool service_request_;
    bool perform_correl_;
    int num_images_;
    int steps_;
    
    // services
    rclcpp::CallbackGroup::SharedPtr cb_group_srv_;
    rclcpp::CallbackGroup::SharedPtr cb_group_client_;
    
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_im_srv_;
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr gpio_client_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr laser_client_;
    rclcpp::Client<ros2_active_stereo_msgs::srv::MoveMotor>::SharedPtr motor_client_;

    //publisher
    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr handshake_images_pub_;

    void send_handshake(int count_to_send) {
        //RCLCPP_INFO(this->get_logger(), "Enviando handshake com %d imagens.", count_to_send);
        auto msg = std_msgs::msg::Int16();
        if (perform_correl_) {
            msg.data = count_to_send;
        } else {
            msg.data = -count_to_send;
        }
        handshake_images_pub_->publish(msg);
        
        service_request_ = false;
        count_ = 0; // Prepara para a próxima

        // Desarma o cão de guarda, se ele existir
        if (watchdog_timer_) {
            watchdog_timer_->cancel();
        }
    }

    void watchdog_timeout_cb() {
        RCLCPP_WARN(this->get_logger(), "Tempo de espera esgotado. Salvamos apenas %d imagens", count_);
        send_handshake(count_); // Envia o que tiver e destrava o Python
    }

    void images_cb(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                   const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) { // salva imagens num arquivo temporario (ja faz parte do pos process))
                
        //RCLCPP_INFO(this->get_logger(), "images_cb started");

        if (!service_request_){
            // RCLCPP_INFO(this->get_logger(), "service_request_ false");
            return;
        }

        try{
            
            cv::Mat left_mat = cv_bridge::toCvShare(left_msg, "mono8")->image;
            cv::Mat right_mat = cv_bridge::toCvShare(right_msg, "mono8")->image;

            if (left_mat.empty() || right_mat.empty()) {
                RCLCPP_WARN(this->get_logger(), "frame vazio recebido");
                return;
            }

            std::string base_path = "/tmp/rrp_stereo/";
            char left_filename[256], right_filename[256];
            
            
            snprintf(left_filename, sizeof(left_filename), "%sleft/L%02d.png", base_path.c_str(), count_+1);
            snprintf(right_filename, sizeof(right_filename), "%sright/R%02d.png", base_path.c_str(), count_+1);

            cv::imwrite(left_filename, left_mat);
            cv::imwrite(right_filename, right_mat);

            count_++;
            //RCLCPP_INFO(this->get_logger(), "[C++] Par %d recebido pelo nó. L_Stamp: %d.%d", count_, left_msg->header.stamp.sec, left_msg->header.stamp.nanosec);

            if (count_ == num_images_-3) {
                RCLCPP_WARN(this->get_logger(), "Sucesso, todas as %d imagens chegaram e foram salvas.", count_);
                send_handshake(count_);
            }

        } catch (cv_bridge::Exception& e){
            RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s error", e.what());
            return;
        }

    }

    void save_images_ssd_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response){
        
        const char* home_dir = std::getenv("HOME");
        if (home_dir == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "Error on finding HOME directory.");
            response->success = false;
            response->message = "HOME not found";
            return;
        }

        std::string ram_path = "/dev/shm/stereo_active/";
        std::string destiny_path = std::string(home_dir) + "/Pictures/stereo_active_backup";
        RCLCPP_INFO(this->get_logger(), "Saving Images ...");

        try{
            std::filesystem::create_directories(destiny_path);
            std::filesystem::copy(ram_path, destiny_path, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
            response->success = true;
            response->message = "Images saved";
            RCLCPP_INFO(this->get_logger(), "Images have been saved");
        }
        catch(const std::filesystem::filesystem_error& e){
            RCLCPP_ERROR(this->get_logger(), "Copy archives error: %s", e.what());
            response->success = false;
            response->message = "Copy archives error";
        }
    }

    void get_images_srv(const std::shared_ptr<std_srvs::srv::SetBool::Request> request, std::shared_ptr<std_srvs::srv::SetBool::Response> response){
        
        count_=0;
        service_request_=true;
        perform_correl_ = request->data;

        float angulo_motor = (steps_ / 2048.0f) * 360.0f;

        // call laser service
        auto laser_request = std::make_shared<std_srvs::srv::SetBool::Request>();
        laser_request->data = true;
        auto future_laser = laser_client_->async_send_request(laser_request); // guarda o pedido do laser_request

        if (future_laser.wait_for(std::chrono::seconds(5)) == std::future_status::ready){ // espera por 5sec o status do laser_request

            auto result_laser = future_laser.get();
            if (result_laser->success){ // caso o laser tenha sido ligado, roda o codigo

                RCLCPP_INFO(this->get_logger(), "Laser turned on!");
                auto move_motor_request = std::make_shared<ros2_active_stereo_msgs::srv::MoveMotor::Request>();

                for(uint8_t i=0; i < num_images_; i++){

                    move_motor_request->angle = angulo_motor;
                    auto future_move_motor = motor_client_->async_send_request(move_motor_request); 
                    if (future_move_motor.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready){ // espera que o motor acabe de se mover

                        auto result_move_motor = future_move_motor.get();
                        if (result_move_motor->success){ 
                            
                            auto trigger_request = std::make_shared<std_srvs::srv::Trigger::Request>();
                            auto future_trigger = gpio_client_->async_send_request(trigger_request);
                            if (future_trigger.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready){

                                auto result_trigger = future_trigger.get();

                                if(result_trigger->success){
                                    //RCLCPP_INFO(this->get_logger(), "[C++] Trigger %d enviado", i+1);
                                    rclcpp::sleep_for(std::chrono::milliseconds(10)); // espera o time exposition da foto + tempo extra
                                }

                            }else{
                                RCLCPP_ERROR(this->get_logger(), "Trigger service call timed out!");
                                response->success = false;
                                response->message = "Trigger failed";
                                return;
                            }
                        }
                    }else{
                        RCLCPP_ERROR(this->get_logger(), "Move_motor service call failed or timed out!");
                        response->success = false;
                        response->message = "Move_motor failed";
                        return;
                    }
                }
            }else{
                RCLCPP_ERROR(this->get_logger(), "Laser service call failed");
                response->success = false;
                response->message = "Laser failed";
                return;
            }
        }
        else{
            RCLCPP_ERROR(this->get_logger(), "Laser service call timed out!");
            response->success = false;
            response->message = "Laser failed";
            return;
        }

        //turn off laser and return motor to initial position
        rclcpp::sleep_for(std::chrono::milliseconds(10));
        laser_request->data = false;
        future_laser = laser_client_->async_send_request(laser_request);

        if (future_laser.wait_for(std::chrono::seconds(5)) == std::future_status::ready){ // espera por 5sec o status do laser_request

           auto result_laser = future_laser.get();
           if (result_laser->success){
                auto move_motor_return_request = std::make_shared<ros2_active_stereo_msgs::srv::MoveMotor::Request>();
                move_motor_return_request->angle = -(steps_*(num_images_)/2048.0f)*360.0f;
                auto future_move_motor_return = motor_client_->async_send_request(move_motor_return_request);
                if (future_move_motor_return.wait_for(std::chrono::seconds(5)) == std::future_status::ready){ 
                     auto result_move_motor_return = future_move_motor_return.get();
                        if (result_move_motor_return->success){ 
                            RCLCPP_INFO(this->get_logger(), "Motor terminou, aguardando as imagens");
                            if (service_request_) { //cria o watchdog apenas se o handshake nao tiver sido enviado
                                watchdog_timer_ = this->create_wall_timer(
                                std::chrono::milliseconds(2500),
                                std::bind(&StereoCorrelProcess::watchdog_timeout_cb, this),
                                cb_group_srv_ // Usa o mesmo grupo para ser thread-safe
                            );
                            }
                            response->success = true;
                            response->message = "Varredura completa";
                        }
                }else{
                    RCLCPP_ERROR(this->get_logger(), "Move_motor_return service call timed out!");
                    response->success = false;
                    response->message = "Motor failed";
                    return;
                }
           }else{
                RCLCPP_ERROR(this->get_logger(), "Laser service call failed!");
                response->success = false;
                response->message = "Laser failed";
                return;
           }

        }else{
            RCLCPP_ERROR(this->get_logger(), "Laser service call timed out!");
            response->success = false;
            response->message = "Laser failed";
            return;
        }

    }
};

}
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoCorrelProcess)