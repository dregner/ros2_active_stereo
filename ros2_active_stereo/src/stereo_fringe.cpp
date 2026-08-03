#include <stereo_fringe.hpp>
#include <chrono>

namespace ros2_active_stereo
{
StereoFringeProcess::StereoFringeProcess(const rclcpp::NodeOptions & options)
: Node("image_project_node", options), n_proj_(0), project_imgs_(false)
{
    // Node params
    this->declare_parameter("monitor_name", "Monitor_1");
    this->declare_parameter("pixel_per_fringe", 128);
    this->declare_parameter("fringe_steps", 4);
    this->declare_parameter("image_color", "blue");
    this->declare_parameter("camera_hz", 20);
    this->declare_parameter("skip_trigger", 3);
    this->declare_parameter("save_path", "/tmp/structured-light");
    this->declare_parameter("save_image", true);
    this->declare_parameter("debug", true);
    this->declare_parameter("trigger_delay", 40);
    // Structured light params to class
    pixel_per_fringe = this->get_parameter("pixel_per_fringe").as_int();
    fringe_steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();
    timer_hz_ = 1000 /  this->get_parameter("camera_hz").as_int();

    // Get Monitor data
    if (!get_screen_resolution(this->get_parameter("monitor_name").as_string())) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get screen resolution");
        return;
    }
    // Construct cv2 window to project
    construct_window();

    // Initiate Fringe and Gray Code generators
    fringe_process_ptr_ = std::make_unique<FringeProcess>(project_resolution_, cv::Size(2448,2048), pixel_per_fringe, fringe_steps);

    // create fringe and graycode images
    fringe_process_ptr_->create_fringe_image();
    fringe_process_ptr_->create_graycode_image(); // Construct images in grayscale by default
    all_imgs_.clear();
    all_imgs_.push_back(black_img_);
    std::vector<cv::Mat> gc_imgs_ = fringe_process_ptr_->get_gc_images(color_);
    std::vector<cv::Mat> fr_imgs_ = fringe_process_ptr_->get_fr_images(color_); // Colors to print patterns (red, blue, green or null for grayscale)
    all_imgs_.insert(all_imgs_.end(), gc_imgs_.begin(), gc_imgs_.end()); // GrayCode first, then Fringe Patterns
    all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end()); // GrayCode first, then Fringe Patterns

    // Callback group to avoid blocking the node with long operations (like cv::imshow)
    timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    srv_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);


    //Subscrbers Quality of Service
    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(2);
    rclcpp::SubscriptionOptions sub_options;    

    // Subscribers
    sub_left_.subscribe(this, "left/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sub_right_.subscribe(this, "right/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(15), sub_left_, sub_right_);
    sync_->registerCallback(std::bind(&StereoFringeProcess::stereo_callback, this, std::placeholders::_1, std::placeholders::_2));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", 10, std::bind(&StereoFringeProcess::camera_info_cb, this, std::placeholders::_1));

    //Publisher 64F images
    pub_abs_left_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/phase_map", 2);
    pub_abs_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/phase_map", 2);
    pub_mod_left_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/modulation_map", 2);
    pub_mod_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/modulation_map", 2);
    // Publihser debug images
    pub_abs_left_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/debug/phase_map", 2);
    pub_abs_right_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/debug/phase_map", 2);


    // Services
    change_image_service_ = this->create_service<std_srvs::srv::SetBool>("image_project",  std::bind(&StereoFringeProcess::project_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );
    process_service_ = this->create_service<std_srvs::srv::Trigger>("phase_process", std::bind(&StereoFringeProcess::process_srv_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );
    save_imgs_service_ = this->create_service<std_srvs::srv::Trigger>("save_image", std::bind(&StereoFringeProcess::save_img_srv_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );
    trigger_client_ = this->create_client<std_srvs::srv::Trigger>("trigger", rmw_qos_profile_default, srv_cb_group_);

    // Timer callback for projection
    timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<long>(timer_hz_)), std::bind(&StereoFringeProcess::project_image_timer_cb, this), timer_cb_group_ );


}

StereoFringeProcess::~StereoFringeProcess() {
    cv::destroyWindow(window_name_);
}

// Process service callback
void StereoFringeProcess::process_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    RCLCPP_INFO(this->get_logger(), "Initate acquisition...");
    n_proj_ = 0; // Garante que a projeção comece do início
    project_imgs_ = true; // Ativa a projeção
    process_ = true;
    RCLCPP_INFO(this->get_logger(), "Project total %d images", all_imgs_.size());
    // Depois de processar, você pode publicar os resultados ou fazer o que for necessário
    response->message = "Initated process";
    response->success = true;
}

// Get Monitor resolution
bool StereoFringeProcess::get_screen_resolution(const std::string& monitor_name)
{
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

// Construct opencv projection window
void StereoFringeProcess::construct_window() 
{

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);  // allow resizing
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);  
}

// Timer callback for projecting images
void StereoFringeProcess::project_image_timer_cb()
{
    int px_f = this->get_parameter("pixel_per_fringe").as_int();
    int steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();
    int delay = this->get_parameter("trigger_delay").as_int();
    // Check if camera info msg has been received
    if (!receive_camera_info_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                             "Waiting for camera info. Sending trigger...");
        send_trigger(); // Encapsulado para limpar o código
        return;
    }

    // Check if parameters have been changed
    if(px_f != pixel_per_fringe || steps != fringe_steps){
        if (project_imgs_) {
            RCLCPP_WARN(this->get_logger(), "Projection parameters changed during active projection. Aborting current projection.");
            project_imgs_ = false;
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Updating projection parameters: pixel_per_fringe=%d, fringe_steps=%d", px_f, steps);
        pixel_per_fringe = px_f;
        fringe_steps = steps;
        // Set new parameters to the pattern generators
        fringe_process_ptr_->FringePattern::set_px_f(pixel_per_fringe);
        fringe_process_ptr_->FringePattern::set_steps(fringe_steps);
        fringe_process_ptr_->GrayCode::set_px_f(pixel_per_fringe);
        // Construct new images
        fringe_process_ptr_->create_fringe_image();
        fringe_process_ptr_->create_graycode_image();

        std::vector<cv::Mat> fr_imgs_ = fringe_process_ptr_->get_fr_images(color_);
        std::vector<cv::Mat> gc_imgs_  = fringe_process_ptr_->get_gc_images(color_);
        all_imgs_.clear();
        all_imgs_.push_back(black_img_);
        all_imgs_.insert(all_imgs_.end(), gc_imgs_.begin(), gc_imgs_.end());
        all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end());
      
        
    }

    // Check if can project and n_proj_ is below projection img number
    if (static_cast<size_t>(n_proj_) < all_imgs_.size() && project_imgs_) {
        cv::imshow(window_name_, all_imgs_[n_proj_]);
        cv::waitKey(1);
    } else  { 
        cv::imshow(window_name_, black_img_);
        cv::waitKey(1);
    }

    // 2. Send trigger if processing and do not receive
    if (process_){
        if(!receive_imgs_){ 
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            send_trigger();
        } 
        else{ 
            receive_imgs_ = false; 
            skip_frames_ = this->get_parameter("skip_trigger").as_int(); 
            n_proj_++;
        }
    }
      
}

void StereoFringeProcess::save_img_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    if(fringe_process_ptr_->save_images(this->get_parameter("save_path").as_string())){
        RCLCPP_INFO(this->get_logger(), "Save images on %s", this->get_parameter("save_path").as_string().c_str());
    }
    else{ RCLCPP_ERROR(this->get_logger(), "Failed to save images");}
}

void StereoFringeProcess::project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    
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

// Left camera info callback
void StereoFringeProcess::camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if(receive_camera_info_) return; // Evita processar múltiplas vezes
    cv::Size cam_res(msg->width, msg->height);
    RCLCPP_INFO(this->get_logger(), "Camera size: %d, %d", msg->width, msg->height);
    fringe_process_ptr_->set_camera_resolution(cam_res);
    RCLCPP_INFO(this->get_logger(), "Received camera info");    
    receive_camera_info_ = true;
}

// Stereo message filter callback (exact time -> tirggered via hardware)
void StereoFringeProcess::stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                                        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) 
{


    // 2. Fluxo de Captura de Padrões
    if (process_) {
        if (skip_frames_ > 0) {
            skip_frames_--;
            return;
        }
        try {
            // Converte imagens usando toCvShare (mais eficiente, sem cópia)
            cv::Mat left = cv_bridge::toCvShare(left_msg, "mono8")->image;
            cv::Mat right = cv_bridge::toCvShare(right_msg, "mono8")->image;
            
            // Armazena no buffer do fringe_process
            if (n_proj_ != 0){
                // RCLCPP_INFO(this->get_logger(), "Processing pattern %d / %zu", n_proj_, all_imgs_.size());
                fringe_process_ptr_->set_images(left, right, (n_proj_-1));
            }
            

            
            if (static_cast<size_t>(n_proj_) >= all_imgs_.size()) {
                process_ = false;
                project_imgs_ = false;
                n_proj_ = 0;
                // RCLCPP_INFO(this->get_logger(), "Sequência completa! Iniciando salvamento...");
                std::vector<cv::Mat> process_result;
                bool debug = this->get_parameter("debug").as_bool();
                process_result = fringe_process_ptr_->calculate_abs_phi_images(debug);
                RCLCPP_INFO(this->get_logger(), "Publishing processed images");
                publish_processed_images(process_result);
            } else{
            // n_proj_++;
            receive_imgs_ = true;
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    } 
    else {
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Idle: Waiting for process_ flag.");
    }
}

// Publish 64 float image to triangulation node
void StereoFringeProcess::publish_processed_images(std::vector<cv::Mat> images)
{
    if (images.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Falha ao publicar: Vetor de imagens incompleto!");
        return;
    }

    // Cria um header sincronizado para o par estéreo
    std_msgs::msg::Header header_left;
    std_msgs::msg::Header header_right;
    header_left.stamp = this->get_clock()->now();
    header_right.stamp = header_left.stamp;
    header_left.frame_id = "Active/left_camera_link"; // Ajuste para o frame de TF real do VORIS
    header_right.frame_id = "Active/right_camera_link"; // Ajuste para o frame de TF real do VORIS

    try {
        // Função lambda auxiliar para normalizar e publicar sem repetir código
        auto publish_normalized = [&](rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub, const cv::Mat& img64, std_msgs::msg::Header hd) {
            if (!img64.empty()) {
                cv::Mat img8u;
                // O Segredo: Escala o Min/Max dinamicamente para caber entre 0 e 255
                cv::normalize(img64, img8u, 0, 255, cv::NORM_MINMAX, CV_8U);
                
                // "mono8" é a string oficial do sensor_msgs para imagens 8UC1
                auto msg = cv_bridge::CvImage(hd, "mono8", img8u).toImageMsg();
                pub->publish(*msg);
            }
        };
        // Converte e publica a Fase Esquerda (phi_l)
        auto msg_phi_l = cv_bridge::CvImage(header_left, "64FC1", images[0]).toImageMsg();
        pub_abs_left_->publish(*msg_phi_l);

        // Converte e publica a Fase Direita (phi_r)
        auto msg_phi_r = cv_bridge::CvImage(header_right, "64FC1", images[1]).toImageMsg();
        pub_abs_right_->publish(*msg_phi_r);

        // Converte e publica a Modulação Esquerda (mod_l)
        publish_normalized(pub_mod_left_, images[2], header_left); // Mod Esq
        publish_normalized(pub_mod_right_, images[3], header_right); // Mod Dir

        if(this->get_parameter("debug").as_bool()){
            //  Publish phase map to visualize
            publish_normalized(pub_abs_left_debug_, images[0], header_left); // Fase Esq
            publish_normalized(pub_abs_right_debug_, images[1], header_right); // Fase Dir
        }

        // RCLCPP_INFO(this->get_logger(), ">>> Mapas 64FC1 publicados com sucesso no ROS 2.");

    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erro fatal no cv_bridge: %s", e.what());
    }
}


// Auxiliar function for trigger
void StereoFringeProcess::send_trigger()
{
    // Check if client is ready to avoid crashing
    if (!trigger_client_->service_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Trigger service not ready!");
        return;
    }

    auto trigger_request = std::make_shared<std_srvs::srv::Trigger::Request>();
    
    // Send request asynchronously with a callback handler, ZERO BLOCKING
    trigger_client_->async_send_request(trigger_request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
            auto response = future.get();
            if (!response->success) {
                RCLCPP_ERROR(this->get_logger(), "Hardware trigger failed!");
            }
        });
}

}

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoFringeProcess)