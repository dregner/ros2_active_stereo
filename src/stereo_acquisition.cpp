#include "stereo_acquisition.hpp"
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

StereoAcquisitionNode::StereoAcquisitionNode(const rclcpp::NodeOptions & options)
: Node("stereo_acquisition_node", options), acquisition_active_(false)
{
    // Parâmetro definindo quantas imagens compõem o set (ex: 4 GrayCode + 4 Franjas = 8)
    total_patterns_ = this->declare_parameter("total_patterns", 8);

    auto qos = rclcpp::SensorDataQoS();

    // 1. Inicializa os assinantes de imagem
    sub_left_.subscribe(this, "left/image_raw", qos.get_rmw_qos_profile());
    sub_right_.subscribe(this, "right/image_raw", qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), sub_left_, sub_right_
    );
    sync_->registerCallback(
        std::bind(&StereoAcquisitionNode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // 2. Inicializa o cliente do serviço do projetor
    projector_client_ = this->create_client<std_srvs::srv::SetBool>("change_image");
    trigger_client_ = this->create_client<std_srvs::srv::Trigger>("trigger");
    acquire_service_ = this->create_service<std_srvs::srv::SetBool>("aquire", std::bind(&StereoAcquisitionNode::acquire_sb, this, std::placeholders::_1, std::placeholders::_2));
    RCLCPP_INFO(this->get_logger(), "Waiting for trigger service..");
    while (!projector_client_->wait_for_service(1s) || !trigger_client_->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "Interrup while waiting service.");
            return;
        }
    }

    stereo_buffer_.reserve(total_patterns_); 


    // 4. Dá o pontapé inicial! Pede a primeira imagem ao projetor
    RCLCPP_INFO(this->get_logger(), "Stereo acquisition ready.");
    // acquisition_active_ = true;
    // request_next_projection(true);
}

void StereoAcquisitionNode::stereo_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
{
    // Se não estivermos ativamente buscando imagens, ignora o fluxo contínuo da câmera
    if (!acquisition_active_) return;

    // Salva o par atual (Zero-Copy)
    stereo_buffer_.push_back(std::make_pair(left_msg, right_msg));
    
    int acquired = stereo_buffer_.size();
    RCLCPP_INFO(this->get_logger(), "Acquired %d/%d images.", acquired, total_patterns_);

    // Verifica se já temos todos os padrões
    if (acquired < total_patterns_) {
        // Pede a próxima imagem (avança o n_proj_ lá no nó do projetor)
        request_next_projection(true);
    } 
    else {
        // Terminamos! Manda o projetor voltar para a tela azul de repouso
        acquisition_active_ = false;
        request_next_projection(false);
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

void StereoAcquisitionNode::acquire_sb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                                       const std::shared_ptr<std_srvs::srv::SetBool::Response> response){
    acquisition_active_ = request->data;
    RCLCPP_INFO(this->get_logger(), "Start acquisition");
    
}

RCLCPP_COMPONENTS_REGISTER_NODE(StereoAcquisitionNode)