#include <stereo_process.hpp>
#include <chrono>


StereoProcessNode::StereoProcessNode(const rclcpp::NodeOptions & options)
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
    this->declare_parameter("save_image", false);
    this->declare_parameter("debug", true);

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
    rclcpp::SubscriptionOptions sub_options;    

    // Subscribers
    sub_left_.subscribe(this, "left/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sub_right_.subscribe(this, "right/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1), sub_left_, sub_right_);
    sync_->registerCallback(std::bind(&StereoProcessNode::stereo_callback, this, std::placeholders::_1, std::placeholders::_2));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", 10, std::bind(&StereoProcessNode::camera_info_cb, this, std::placeholders::_1));

    //Publisher 64F images
    pub_abs_left_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/absolute_image", 10);
    pub_abs_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/absolute_image", 10);
    pub_mod_left_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/modulation_image", 10);
    pub_mod_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/modulation_image", 10);
    // Publihser debug images
    pub_abs_left_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/debug/absolute_image", 10);
    pub_abs_right_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/debug/absolute_image", 10);
    pub_mod_left_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/left/debug/modulation_image", 10);
    pub_mod_right_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/debug/modulation_image", 10);

    // Services
    change_image_service_ = this->create_service<std_srvs::srv::SetBool>("image_project",  std::bind(&StereoProcessNode::project_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );
    process_service_ = this->create_service<std_srvs::srv::Trigger>("process", std::bind(&StereoProcessNode::process_srv_cb, this, std::placeholders::_1, std::placeholders::_2), rmw_qos_profile_default );
    trigger_client_ = this->create_client<std_srvs::srv::Trigger>("trigger", rmw_qos_profile_default, srv_cb_group_);

    // Timer callback for projection
    timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<long>(timer_hz_)), std::bind(&StereoProcessNode::project_image_timer_cb, this), timer_cb_group_ );


}

StereoProcessNode::~StereoProcessNode() {
    cv::destroyWindow(window_name_);
}

// Process service callback
void StereoProcessNode::process_srv_cb(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                    const std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    RCLCPP_INFO(this->get_logger(), "Initate acquisition...");
    n_proj_ = 0; // Garante que a projeção comece do início
    project_imgs_ = true; // Ativa a projeção
    process_ = true;
    // Depois de processar, você pode publicar os resultados ou fazer o que for necessário
    response->message = "Initated process";
    response->success = true;
}

// Get Monitor resolution
bool StereoProcessNode::get_screen_resolution(const std::string& monitor_name)
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
void StereoProcessNode::construct_window() 
{

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);  // allow resizing
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);  
}

/* Timer callback for projecting images */
void StereoProcessNode::project_image_timer_cb()
{
    int px_f = this->get_parameter("pixel_per_fringe").as_int();
    int steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();

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
        cv::waitKey(10);
    } else { 
        cv::imshow(window_name_, black_img_);
        cv::waitKey(10);
    }

    // 2. Send trigger if processing and do not receive
    if (process_){
        if(!receive_imgs_){ send_trigger();} 
        else{ 
            receive_imgs_ = false; 
            skip_frames_ = this->get_parameter("skip_trigger").as_int(); 
            n_proj_++;
        }
    }
      
    if(this->get_parameter("save_image").as_bool()){
        if(fringe_process_ptr_->save_images(this->get_parameter("save_path").as_string())){
            RCLCPP_INFO(this->get_logger(), "Save images on %s", this->get_parameter("save_path").as_string().c_str());
        }else{ RCLCPP_ERROR(this->get_logger(), "Failed to save images");}
    }
}

/*Bool Project service callback, False: reset, true: start process*/
void StereoProcessNode::project_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
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
void StereoProcessNode::camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if(receive_camera_info_) return; // Evita processar múltiplas vezes
    cv::Size cam_res(msg->width, msg->height);
    RCLCPP_INFO(this->get_logger(), "Camera size: %d, %d", msg->width, msg->height);
    fringe_process_ptr_->set_camera_resolution(cam_res);
    RCLCPP_INFO(this->get_logger(), "Received camera info");    
    receive_camera_info_ = true;
}

// Stereo message filter callback (exact time -> tirggered via hardware)
void StereoProcessNode::stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                                        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) 
{


    // 2. Fluxo de Captura de Padrões
    if (process_ && !receive_imgs_) {
        if (skip_frames_ > 0) {
            skip_frames_--;
            return;
        }
        try {
            // Converte imagens usando toCvShare (mais eficiente, sem cópia)
            cv::Mat left = cv_bridge::toCvShare(left_msg, "mono8")->image;
            cv::Mat right = cv_bridge::toCvShare(right_msg, "mono8")->image;

            RCLCPP_INFO(this->get_logger(), "Processing pattern %d / %zu", 
                        n_proj_, all_imgs_.size());

            // Armazena no buffer do fringe_process
            if (n_proj_ != 0){
                fringe_process_ptr_->set_images(left, right, (n_proj_-1));
            }
            receive_imgs_ = true;
            
            trigger_timer_ = 0;

            if (static_cast<size_t>(n_proj_) >= all_imgs_.size()) {
                process_ = false;
                project_imgs_ = false;
                n_proj_ = 0;
                RCLCPP_INFO(this->get_logger(), "Sequência completa! Iniciando salvamento...");
                std::vector<cv::Mat> process_result;
                bool debug = this->get_parameter("debug").as_bool();
                process_result = fringe_process_ptr_->calculate_abs_phi_images(debug);
                RCLCPP_INFO(this->get_logger(), "Publishing processed images");
                publish_processed_images(process_result);
                if (debug){
                    publish_processed_debug_images(process_result);
                }
            } 

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    } 
    else {
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Idle: Waiting for project_imgs_ flag.");
    }
}

// Publish 64 float image to triangulation node
void StereoProcessNode::publish_processed_images(std::vector<cv::Mat> images)
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
        // Converte e publica a Fase Esquerda (phi_l)
        auto msg_phi_l = cv_bridge::CvImage(header_left, "64FC1", images[0]).toImageMsg();
        pub_abs_left_->publish(*msg_phi_l);

        // Converte e publica a Fase Direita (phi_r)
        auto msg_phi_r = cv_bridge::CvImage(header_right, "64FC1", images[1]).toImageMsg();
        pub_abs_right_->publish(*msg_phi_r);

        // Converte e publica a Modulação Esquerda (mod_l)
        auto msg_mod_l = cv_bridge::CvImage(header_left, "64FC1", images[2]).toImageMsg();
        pub_mod_left_->publish(*msg_mod_l);

        // Converte e publica a Modulação Direita (mod_r)
        auto msg_mod_r = cv_bridge::CvImage(header_right, "64FC1", images[3]).toImageMsg();
        pub_mod_right_->publish(*msg_mod_r);

        RCLCPP_INFO(this->get_logger(), ">>> Mapas 64FC1 publicados com sucesso no ROS 2.");

    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erro fatal no cv_bridge: %s", e.what());
    }
}

// Publish debug image to visualize phase and modulation maps
void StereoProcessNode::publish_processed_debug_images(std::vector<cv::Mat> images)
{
    if (images.size() < 4) {
        RCLCPP_WARN(this->get_logger(), "Falha ao publicar 8U: Vetor de imagens incompleto!");
        return;
    }

    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = "camera_link"; 

    try {
        // Função lambda auxiliar para normalizar e publicar sem repetir código
        auto publish_normalized = [&](rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub, const cv::Mat& img64) {
            if (!img64.empty()) {
                cv::Mat img8u;
                // O Segredo: Escala o Min/Max dinamicamente para caber entre 0 e 255
                cv::normalize(img64, img8u, 0, 255, cv::NORM_MINMAX, CV_8U);
                
                // "mono8" é a string oficial do sensor_msgs para imagens 8UC1
                auto msg = cv_bridge::CvImage(header, "mono8", img8u).toImageMsg();
                pub->publish(*msg);
            }
        };

        // Dispara as publicações
        publish_normalized(pub_abs_left_debug_, images[0]); // Fase Esq
        publish_normalized(pub_abs_right_debug_, images[1]); // Fase Dir
        publish_normalized(pub_mod_left_debug_, images[2]); // Mod Esq
        publish_normalized(pub_mod_right_debug_, images[3]); // Mod Dir

        RCLCPP_INFO(this->get_logger(), ">>> Mapas 8UC1 (Visualizacao) publicados com sucesso.");

    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erro no cv_bridge (8U): %s", e.what());
    }
}

// Auxiliar function for trigger
void StereoProcessNode::send_trigger()
{
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    trigger_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
            this->trigger_callback(future);
        }
    );
}

// Trigger callback
void StereoProcessNode::trigger_callback(rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future)
{
    try {
        // 1. Tenta obter o resultado do "future"
        auto response = future.get();

        if (response->success) {
            // Opcional: Log de sucesso (use DEBUG para não poluir o terminal)
            RCLCPP_DEBUG(this->get_logger(), "Trigger enviado com sucesso: %s", response->message.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "O serviço de Trigger falhou: %s", response->message.c_str());
            
            // Estratégia de erro: se falhar, talvez você queira resetar o n_proj_
            // ou tentar enviar o trigger novamente.
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exceção ao receber resposta do serviço: %s", e.what());
    }
}


RCLCPP_COMPONENTS_REGISTER_NODE(StereoProcessNode)