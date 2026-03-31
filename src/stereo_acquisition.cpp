#include "stereo_acquisition.hpp"
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

StereoAcquisitionNode::StereoAcquisitionNode(const rclcpp::NodeOptions & options)
: Node("stereo_acquisition_node", options), acquisition_active_(false)
{

    auto qos = rclcpp::SensorDataQoS();
    rclcpp::SubscriptionOptions sub_options;    

    sub_left_.subscribe(this, "left/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sub_right_.subscribe(this, "right/image_raw", qos.get_rmw_qos_profile(), sub_options);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), sub_left_, sub_right_
    );
    sync_->registerCallback(
        std::bind(&StereoAcquisitionNode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // Configuração dos clientes com o grupo de callback
    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    projector_client_ = this->create_client<std_srvs::srv::SetBool>("project", qos.get_rmw_qos_profile(), cb_group_);
    trigger_client_ = this->create_client<std_srvs::srv::Trigger>("trigger", qos.get_rmw_qos_profile(), cb_group_);
    acquire_service_ = this->create_service<std_srvs::srv::SetBool>("acquire", 
                                                                    std::bind(&StereoAcquisitionNode::acquire_cb, this,
                                                                         std::placeholders::_1, std::placeholders::_2),
                                                                    rmw_qos_profile_default);

    n_imgs_sub_ = this->create_subscription<std_msgs::msg::Int8>(
        "total_fringe_images", 10, 
        [this](const std_msgs::msg::Int8::SharedPtr msg) {
            total_patterns_ = msg->data;
            RCLCPP_INFO(this->get_logger(), "Total patterns to acquire: %d", total_patterns_);
        }
    );

    idx_proj_pub_ = this->create_publisher<std_msgs::msg::Int8>("fringe_index", 10);
                                                             


    stereo_buffer_.reserve(total_patterns_); 
    timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&StereoAcquisitionNode::timer_callback, this), timer_cb_group_);

    RCLCPP_INFO(this->get_logger(), "Stereo acquisition ready.");
}

void StereoAcquisitionNode::timer_callback() {
    // Publica o número de imagens adquiridas para o nó do projetor
    // RCLCPP_INFO(this->get_logger(), "Timer CB");
    if(acquisition_active_){
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = true; // Sinaliza para o projetor avançar a proje
        auto result_future = projector_client_->async_send_request(request);
        std::future_status status = result_future.wait_for(std::chrono::milliseconds(200));
        if (status == std::future_status::ready) {
            auto response = result_future.get();
            if (!response->success) {
                RCLCPP_ERROR(this->get_logger(), "Failed to request next projection: %s", response->message.c_str());
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Timeout while waiting for projector response.");
            auto trigger_request = std::make_shared<std_srvs::srv::Trigger::Request>();
            auto trigger_response_future = trigger_client_->async_send_request(trigger_request);
            std::future_status trigger_status = trigger_response_future.wait_for(std::chrono::milliseconds(200));
            if (trigger_status == std::future_status::ready) {
                auto trigger_response = trigger_response_future.get();
                if (!trigger_response->success) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to trigger next projection: %s", trigger_response->message.c_str());
                }
            } else {
                    RCLCPP_ERROR(this->get_logger(), "Timeout while waiting for trigger response.");
                }
        }
    }
}
void StereoAcquisitionNode::stereo_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
{
    // Se não estivermos ativamente buscando imagens, ignora o fluxo contínuo da câmera
    if (!acquisition_active_) {
        auto message = std_msgs::msg::Int8();
        message.data = 0;
        idx_proj_pub_->publish(message);
        return;
    }

    // Salva o par atual (Zero-Copy)
    stereo_buffer_.push_back(std::make_pair(left_msg, right_msg));
    
    int acquired = stereo_buffer_.size();
    RCLCPP_INFO(this->get_logger(), "Acquired %d/%d images.", acquired, total_patterns_);
        // Publish index
    auto message = std_msgs::msg::Int8();
    message.data = static_cast<int8_t>(acquired);
    idx_proj_pub_->publish(message);
    // Verifica se já temos todos os padrões
    if (acquired > total_patterns_) {
        // Terminamos! Manda o projetor voltar para a tela azul de repouso
        acquisition_active_ = false;
        RCLCPP_INFO(this->get_logger(), "Aquisicao completa. Projetor em repouso. Iniciando LibTorch...");
    }
}

void StereoAcquisitionNode::request_next_projection(bool turn_on)
{
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = turn_on;

    projector_client_->async_send_request(
        request,
        [this, turn_on](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
            if (turn_on && future.get()->success) {
                // Pausa para estabilizacao fisica do hardware de projecao
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                
                auto trig_req = std::make_shared<std_srvs::srv::Trigger::Request>();
                trigger_client_->async_send_request(trig_req);
            }
        }
    );
}

void StereoAcquisitionNode::acquire_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                                       const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    if(request->data){
        RCLCPP_INFO(this->get_logger(), "Starting acquisition...");
    }
    acquisition_active_ = request->data;
    response->success = true;
    response->message = "Acquisition started.";
    
}

RCLCPP_COMPONENTS_REGISTER_NODE(StereoAcquisitionNode)