#include <image_project.hpp>


ImageProjectNode::ImageProjectNode(const rclcpp::NodeOptions & options)
: Node("image_project_node", options), n_proj_(0), project_imgs_(false)
{
    this->declare_parameter("monitor_name", "Monitor_0");
    this->declare_parameter("pixel_per_frigne", 16);
    this->declare_parameter("fringe_steps", 4);
    this->declare_parameter("image_color", "blue");

    pixel_per_fringe = this->get_parameter("pixel_per_frigne").as_int();
    fringe_steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();

    // Get Monitor data
    get_screen_resolution(this->get_parameter("monitor_name").as_string());
    // Construct cv2 window to project
    construct_window();

    // Instanciação dinâmica logo após ler os parâmetros do ROS 2
    fringe_ptr_ = std::make_unique<FringePattern>(project_resolution_, pixel_per_fringe, fringe_steps);
    graycode_ptr_ = std::make_unique<GrayCode>(project_resolution_, 0, pixel_per_fringe);

    fringe_ptr_->create_fringe_image();
    graycode_ptr_->create_graycode_image();
    // std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_color_fr_image(color_);
    // all_imgs_ = graycode_ptr_->get_color_gc_image(color_);
    std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_fr_images();
    all_imgs_ = graycode_ptr_->get_gc_images();
    all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end());

    change_image_service_ = this->create_service<std_srvs::srv::SetBool>("projector", std::bind(&ImageProjectNode::project_cb, this, std::placeholders::_1, std::placeholders::_2));
    // image_idx_pub = this->create_publisher<std_msgs::msg::Int8>("project_idx", 10);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&ImageProjectNode::project_image_timer_cb, this));

}

ImageProjectNode::~ImageProjectNode() {
    cv::destroyWindow(window_name_);
}

void ImageProjectNode::get_screen_resolution(const std::string& monitor_name) {
    auto monitors = get_monitors();

    for (const auto& monitor : monitors) {
        if (monitor.name == monitor_name) {
            project_resolution_.width = monitor.width;
            project_resolution_.height = monitor.height;
            this->window_position_ = {monitor.x, monitor.y};
            RCLCPP_INFO(this->get_logger(),
                        "Monitor %s resolution: %dx%d, position: %dx%d",
                        monitor_name.c_str(),
                        project_resolution_.width, project_resolution_.height,
                        monitor.x, monitor.y);
            black_img_ = cv::Mat::ones(project_resolution_.height, project_resolution_.width, CV_8UC1)*255;

            return;
        }
    }

    RCLCPP_WARN(this->get_logger(), "Monitor '%s' not found. Using default values.", monitor_name.c_str());
}

void ImageProjectNode::construct_window() {

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);  // allow resizing
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);  
}

void ImageProjectNode::project_image_timer_cb(){
    int px_f = this->get_parameter("pixel_per_frigne").as_int();
    int steps = this->get_parameter("fringe_steps").as_int();
    color_ = this->get_parameter("image_color").as_string();
    std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_fr_images();

    if(px_f != pixel_per_fringe || steps != fringe_steps){
        pixel_per_fringe = px_f;
        fringe_steps = steps;
        fringe_ptr_->create_fringe_image();
        graycode_ptr_->create_graycode_image();
        std::vector<cv::Mat> fr_imgs_ = fringe_ptr_->get_color_fr_image(color_);
        all_imgs_ = graycode_ptr_->get_color_gc_image(color_);
        all_imgs_.insert(all_imgs_.end(), fr_imgs_.begin(), fr_imgs_.end());
    }

    if (n_proj_ < all_imgs_.size() && project_imgs_){
        // RCLCPP_INFO(this->get_logger(), "Project %d", n_proj_);
        cv::imshow(window_name_, fr_imgs_[n_proj_]);
    }
    // else{
    //     cv::imshow(window_name_, black_img_);
    // }
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
        } else {
            // Próximos chamados: câmera requisitou o próximo quadro
            RCLCPP_INFO(this->get_logger(), "Projection: %d", n_proj_);
            n_proj_++;
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