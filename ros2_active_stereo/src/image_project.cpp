#include <image_project.hpp>


ImageProjectNode::ImageProjectNode(const rclcpp::NodeOptions & options)
: Node("image_project_node", options), n_proj_(0), project_imgs_(false)
{
    this->declare_parameter("monitor_name", "Monitor_1");
    this->declare_parameter("pixel_per_frigne", 128);
    this->declare_parameter("fringe_steps", 4);
    this->declare_parameter("image_color", "blue");

    pixel_per_fringe = this->get_parameter("pixel_per_frigne").as_int();
    fringe_steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();

    // Get Monitor data
    if (!get_screen_resolution(this->get_parameter("monitor_name").as_string())) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get screen resolution");
        return;
    }
    // Construct cv2 window to project
    construct_window();

    // Initiate Fringe and Gray Code generators
    fringe_ptr_ = std::make_unique<FringePattern>(project_resolution_, pixel_per_fringe, fringe_steps);
    graycode_ptr_ = std::make_unique<GrayCode>(project_resolution_, 0, pixel_per_fringe);

    fringe_ptr_->create_fringe_image();
    graycode_ptr_->create_graycode_image(); // Construct images in grayscale by default

    all_imgs_ = graycode_ptr_->get_gc_images(color_);
    std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_fr_images(color_); // Colors to print patterns (red, blue, green or null for grayscale)
    all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end()); // GrayCode first, then Fringe Patterns

    // Callback group to avoid blocking the node with long operations (like cv::imshow)
    timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    
    // Image info publisher
    n_imgs_pub_ = this->create_publisher<std_msgs::msg::Int8>("total_fringe_images", 10);
    idx_proj_sub_ = this->create_subscription<std_msgs::msg::Int8>(
        "fringe_index", 10, 
        [this](const std_msgs::msg::Int8::SharedPtr msg) {
            if(project_imgs_){
            n_proj_ = msg->data;
            // RCLCPP_INFO(this->get_logger(), "Total patterns to acquire: %d", n_proj_);
            }
        }
    );

    change_image_service_ = this->create_service<std_srvs::srv::SetBool>(
        "project",  std::bind(&ImageProjectNode::project_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );

    // O timer também entra no grupo para não bloquear o nó
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10), 
        std::bind(&ImageProjectNode::project_image_timer_cb, this),
        timer_cb_group_
    );

    // For debug only
    left_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/left/image_raw", 10);
    right_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/right/image_raw", 10);
    trigger_service_ = this->create_service<std_srvs::srv::Trigger>("trigger", std::bind(&ImageProjectNode::trigger_cb, this, std::placeholders::_1, std::placeholders::_2));

}

ImageProjectNode::~ImageProjectNode() {
    cv::destroyWindow(window_name_);
}

void ImageProjectNode::trigger_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    // Simula a resposta do projetor para avançar a projeção
    response->success = true;
    response->message = "Triggered next projection.";
}

bool ImageProjectNode::get_screen_resolution(const std::string& monitor_name) {
    auto monitors = get_monitors();

        // 1. Imprime TODOS os monitores encontrados para debug
        RCLCPP_INFO(this->get_logger(), "Encontrados %zu monitores conectados:", monitors.size());
        for (const auto& monitor : monitors) {
            RCLCPP_INFO(this->get_logger(),
                            " -> Monitor %s: resolucao %dx%d, posicao %dx%d",
                            monitor.name.c_str(),
                            monitor.width, monitor.height,
                            monitor.x, monitor.y);
        }

        
        for (const auto& monitor : monitors) {
            if (monitor.name == monitor_name) {
                project_resolution_.width = monitor.width;
                project_resolution_.height = monitor.height;
                this->window_position_ = {monitor.x, monitor.y};
                
                RCLCPP_INFO(this->get_logger(),
                            "Monitor '%s' selecionado com sucesso!",
                            monitor_name.c_str());
                            
                black_img_ = cv::Mat::zeros(project_resolution_.height, project_resolution_.width, CV_8UC1);
                return true;
            }
        }

    RCLCPP_ERROR(this->get_logger(), "Monitor '%s' not found", monitor_name.c_str());
    return false;
}

void ImageProjectNode::construct_window() {

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);  // allow resizing
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);  
}

void ImageProjectNode::project_image_timer_cb(){
    int px_f = this->get_parameter("pixel_per_frigne").as_int();
    int steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();

    if(px_f != pixel_per_fringe || steps != fringe_steps){
        pixel_per_fringe = px_f;
        fringe_steps = steps;
        // Set new parameters to the pattern generators
        fringe_ptr_->set_px_f(pixel_per_fringe);
        fringe_ptr_->set_steps(fringe_steps);
        graycode_ptr_->set_px_f(pixel_per_fringe);
        // Construct new images
        fringe_ptr_->create_fringe_image();
        graycode_ptr_->create_graycode_image();

        std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_fr_images(color_);
        all_imgs_ = graycode_ptr_->get_gc_images(color_);
        all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end());
        
    }

    if (static_cast<size_t>(n_proj_) < all_imgs_.size() && project_imgs_){
        cv::imshow(window_name_, all_imgs_[n_proj_]);
        // Detecta o formato dinamicamente (útil para testar o Gray Code Cinza vs Azul)
        std::string encoding = (all_imgs_[n_proj_].channels() == 3) ? "bgr8" : "mono8";

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_link"; // Frame padrão para TF

        try {
            // Converte as matrizes OpenCV para mensagens ROS 2
            sensor_msgs::msg::Image::SharedPtr left_msg = cv_bridge::CvImage(header, encoding, all_imgs_[n_proj_]).toImageMsg();
            sensor_msgs::msg::Image::SharedPtr right_msg = cv_bridge::CvImage(header, encoding, all_imgs_[n_proj_]).toImageMsg();

            // Publica nos tópicos
            left_pub_->publish(*left_msg);
            right_pub_->publish(*right_msg);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    else{
        cv::imshow(window_name_, black_img_);
    }

    auto message = std_msgs::msg::Int8();
    message.data = static_cast<int8_t>(all_imgs_.size());
    n_imgs_pub_->publish(message);

    cv::waitKey(10);

}

void ImageProjectNode::project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         const std::shared_ptr<std_srvs::srv::SetBool::Response> response){
    if (request->data) {
        if (!project_imgs_) {
            // Primeiro chamado: inicia a sequência na imagem zero
            n_proj_ = 0;
            project_imgs_ = true;
            RCLCPP_INFO(this->get_logger(), "Start structured light projection.");
            RCLCPP_INFO(this->get_logger(), "Projection: %d", n_proj_);
        } else {
            // Próximos chamados: câmera requisitou o próximo quadro
            n_proj_++;
            RCLCPP_INFO(this->get_logger(), "Projection: %d", n_proj_);
            if (static_cast<std::size_t>(n_proj_) >= all_imgs_.size()) {
                project_imgs_ = false; // Acabaram as imagens, apaga a tela
                RCLCPP_INFO(this->get_logger(), "Finished images.");
            } else {
                RCLCPP_DEBUG(this->get_logger(), "Project index %d", n_proj_);
            }
        }
    } else {
        // Comando explícito para abortar a iluminação
        project_imgs_ = false;
        n_proj_ = 0;
        RCLCPP_INFO(this->get_logger(), "Aborted projection.");
    }
    
    response->success = true;
}
RCLCPP_COMPONENTS_REGISTER_NODE(ImageProjectNode)