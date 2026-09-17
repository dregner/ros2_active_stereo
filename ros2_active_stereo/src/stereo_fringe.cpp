#include <stereo_fringe.hpp>
#include <chrono>
#include <mutex>
#include <algorithm>

namespace ros2_active_stereo
{

// ── Constructor ───────────────────────────────────────────────────────────────

StereoFringeProcess::StereoFringeProcess(const rclcpp::NodeOptions & options)
: Node("image_project_node", options)
{
    // ── Parameters ────────────────────────────────────────────────────────
    this->declare_parameter("monitor_name",      "Monitor_1");
    this->declare_parameter("pixel_per_fringe",  128);
    this->declare_parameter("fringe_steps",      4);
    this->declare_parameter("image_color",       "blue");
    this->declare_parameter("camera_hz",         30);
    this->declare_parameter("save_path",         "/tmp/structured-light");
    this->declare_parameter("save_image",        false);
    this->declare_parameter("debug",             false);
    this->declare_parameter("settle_ms",         22);

    pixel_per_fringe_ = this->get_parameter("pixel_per_fringe").as_int();
    fringe_steps_     = this->get_parameter("fringe_steps").as_int();
    color_            = this->get_parameter("image_color").as_string();
    settle_ms_        = this->get_parameter("settle_ms").as_int();

    // ── Screen & Window Setup ─────────────────────────────────────────────
    if (!get_screen_resolution(this->get_parameter("monitor_name").as_string())) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get screen resolution");
        return;
    }
    construct_window();

    // ── Pattern Generation ────────────────────────────────────────────────
    fringe_process_ptr_ = std::make_unique<FringeProcess>(
        project_resolution_,
        cv::Size(2448, 2048),
        pixel_per_fringe_,
        fringe_steps_);

    rebuild_patterns();

    // ── Callback Groups ───────────────────────────────────────────────────
    // display_cb_group_: pumps OpenCV event loop (wall timer)
    // srv_cb_group_    : services + trigger client (MutuallyExclusive so service
    //                    callbacks never run concurrently with each other)
    // cam_cb_group_    : camera image subscribers (Reentrant so left & right can
    //                    arrive and be pushed to their queues simultaneously)
    display_cb_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    srv_cb_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    cam_cb_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);

    // ── Subscribers ───────────────────────────────────────────────────────
    // Independent left/right subscriptions – no synchronizer needed.
    // Each callback just pushes the decoded cv::Mat into its queue.
    // run_acquisition() pops one frame per camera per pattern step.
    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(4);   // small buffer; we consume frames as fast as they arrive

    rclcpp::SubscriptionOptions cam_opts;
    cam_opts.callback_group = cam_cb_group_;

    sub_left_  = this->create_subscription<sensor_msgs::msg::Image>(
        "left/image_raw", qos,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
            left_image_cb(msg);
        }, cam_opts);

    sub_right_ = this->create_subscription<sensor_msgs::msg::Image>(
        "right/image_raw", qos,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
            right_image_cb(msg);
        }, cam_opts);

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", 10,
        std::bind(&StereoFringeProcess::camera_info_cb, this, std::placeholders::_1));

    // ── Publishers ────────────────────────────────────────────────────────
    pub_abs_left_  = this->create_publisher<sensor_msgs::msg::Image>("sync/left/phase_map",       2);
    pub_abs_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/phase_map",      2);
    pub_mod_left_  = this->create_publisher<sensor_msgs::msg::Image>("sync/left/modulation_map",  2);
    pub_mod_right_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/modulation_map", 2);

    pub_abs_left_debug_  = this->create_publisher<sensor_msgs::msg::Image>("sync/left/debug/phase_map",  2);
    pub_abs_right_debug_ = this->create_publisher<sensor_msgs::msg::Image>("sync/right/debug/phase_map", 2);

    scan_done_pub_ = this->create_publisher<std_msgs::msg::String>("fringe_status", 10);

    // ── Services / Clients ────────────────────────────────────────────────
    change_image_service_ = this->create_service<std_srvs::srv::SetBool>(
        "image_project",
        std::bind(&StereoFringeProcess::project_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        rclcpp::ServicesQoS(), srv_cb_group_);

    process_service_ = this->create_service<std_srvs::srv::Trigger>(
        "phase_process",
        std::bind(&StereoFringeProcess::process_srv_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        rclcpp::ServicesQoS(), srv_cb_group_);

    save_imgs_service_ = this->create_service<std_srvs::srv::Trigger>(
        "save_image",
        std::bind(&StereoFringeProcess::save_img_srv_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        rclcpp::ServicesQoS(), srv_cb_group_);

    trigger_client_ = this->create_client<std_srvs::srv::Trigger>(
        "trigger", rclcpp::ServicesQoS(), srv_cb_group_);

    // ── Display Timer ─────────────────────────────────────────────────────
    display_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&StereoFringeProcess::display_timer_cb, this),
        display_cb_group_);
}

// ── Destructor ────────────────────────────────────────────────────────────────

StereoFringeProcess::~StereoFringeProcess() {
    acquiring_ = false;
    // Wake any thread blocked in pop_left / pop_right so it can exit cleanly
    left_cv_.notify_all();
    right_cv_.notify_all();
    if (acquisition_thread_.joinable()) {
        acquisition_thread_.join();
    }
    cv::destroyWindow(window_name_);
}

// ── Screen / window ───────────────────────────────────────────────────────────

bool StereoFringeProcess::get_screen_resolution(const std::string& monitor_name)
{
    auto monitors = get_monitors();
    RCLCPP_INFO(this->get_logger(), "Found %zu monitor(s):", monitors.size());
    for (const auto& m : monitors) {
        RCLCPP_INFO(this->get_logger(), "  -> %s: %dx%d @ (%d,%d)",
                    m.name.c_str(), m.width, m.height, m.x, m.y);
    }
    for (const auto& m : monitors) {
        if (m.name == monitor_name) {
            project_resolution_ = {m.width, m.height};
            window_position_    = {m.x, m.y};
            black_img_ = cv::Mat::zeros(m.height, m.width, CV_8UC1);
            RCLCPP_INFO(this->get_logger(), "Selected projector monitor '%s'", monitor_name.c_str());
            return true;
        }
    }
    RCLCPP_ERROR(this->get_logger(), "Monitor '%s' not found", monitor_name.c_str());
    return false;
}

void StereoFringeProcess::construct_window()
{
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::moveWindow(window_name_, window_position_.first, window_position_.second);
    cv::setWindowProperty(window_name_, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    cv::imshow(window_name_, black_img_);
    cv::waitKey(1);
}

void StereoFringeProcess::rebuild_patterns()
{
    fringe_process_ptr_->FringePattern::set_px_f(pixel_per_fringe_);
    fringe_process_ptr_->FringePattern::set_steps(fringe_steps_);
    fringe_process_ptr_->GrayCode::set_px_f(pixel_per_fringe_);
    fringe_process_ptr_->create_fringe_image();
    fringe_process_ptr_->create_graycode_image();

    all_imgs_.clear();
    auto gc_imgs = fringe_process_ptr_->get_gc_images(color_);
    auto fr_imgs = fringe_process_ptr_->get_fr_images(color_);
    all_imgs_.insert(all_imgs_.end(), gc_imgs.begin(), gc_imgs.end());
    all_imgs_.insert(all_imgs_.end(), fr_imgs.begin(), fr_imgs.end());

    RCLCPP_INFO(this->get_logger(), "Fringe patterns ready: %zu total (%zu GrayCode + %d Fringe)",
                all_imgs_.size(), gc_imgs.size(), fringe_steps_);
}

// ── Display timer (keeps OpenCV event loop alive while idle) ──────────────────

void StereoFringeProcess::display_timer_cb()
{
    if (!acquiring_) {
        cv::waitKey(1);
    }
}

// ── Camera image callbacks – push to independent queues ───────────────────────

void StereoFringeProcess::left_image_cb(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    if (!acquiring_) return;
    try {
        cv::Mat img = cv_bridge::toCvShare(msg, "mono8")->image.clone();
        {
            std::lock_guard<std::mutex> lk(left_mtx_);
            left_queue_.push(std::move(img));
        }
        left_cv_.notify_one();
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "left_image_cb cv_bridge: %s", e.what());
    }
}

void StereoFringeProcess::right_image_cb(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    if (!acquiring_) return;
    try {
        cv::Mat img = cv_bridge::toCvShare(msg, "mono8")->image.clone();
        {
            std::lock_guard<std::mutex> lk(right_mtx_);
            right_queue_.push(std::move(img));
        }
        right_cv_.notify_one();
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "right_image_cb cv_bridge: %s", e.what());
    }
}

// Blocking pop with timeout – mirrors stereoSystem.triggerAndReceive() semantics
cv::Mat StereoFringeProcess::pop_left(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lk(left_mtx_);
    if (!left_cv_.wait_for(lk, timeout,
            [this]{ return !left_queue_.empty() || !acquiring_; })) {
        return {};   // timeout
    }
    if (left_queue_.empty()) return {};  // acquiring_ went false
    cv::Mat img = std::move(left_queue_.front());
    left_queue_.pop();
    return img;
}

cv::Mat StereoFringeProcess::pop_right(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lk(right_mtx_);
    if (!right_cv_.wait_for(lk, timeout,
            [this]{ return !right_queue_.empty() || !acquiring_; })) {
        return {};
    }
    if (right_queue_.empty()) return {};
    cv::Mat img = std::move(right_queue_.front());
    right_queue_.pop();
    return img;
}

// ── Camera info ───────────────────────────────────────────────────────────────

void StereoFringeProcess::camera_info_cb(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if (receive_camera_info_) return;
    camera_resolution_ = cv::Size(msg->width, msg->height);
    fringe_process_ptr_->set_camera_resolution(camera_resolution_);
    RCLCPP_INFO(this->get_logger(), "Camera resolution registered: %ux%u", msg->width, msg->height);
    receive_camera_info_ = true;
}

// ── /phase_process service ────────────────────────────────────────────────────

void StereoFringeProcess::process_srv_cb(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    if (acquiring_) {
        response->success = false;
        response->message = "Acquisition already in progress";
        return;
    }

    if (!trigger_client_->service_is_ready()) {
        response->success = false;
        response->message = "Trigger service not ready";
        return;
    }

    if (acquisition_thread_.joinable()) {
        acquisition_thread_.join();
    }

    // Re-read parameters so they can be changed without restarting the node
    int px_f  = this->get_parameter("pixel_per_fringe").as_int();
    int steps = this->get_parameter("fringe_steps").as_int();
    settle_ms_ = this->get_parameter("settle_ms").as_int();
    color_     = this->get_parameter("image_color").as_string();

    if (px_f != pixel_per_fringe_ || steps != fringe_steps_) {
        pixel_per_fringe_ = px_f;
        fringe_steps_     = steps;
        rebuild_patterns();
    }

    // Drain stale frames from both queues before starting
    { std::lock_guard<std::mutex> lk(left_mtx_);  while (!left_queue_.empty())  left_queue_.pop(); }
    { std::lock_guard<std::mutex> lk(right_mtx_); while (!right_queue_.empty()) right_queue_.pop(); }

    acquiring_ = true;
    response->success = true;
    response->message = "Acquisition started";

    auto status = std_msgs::msg::String();
    status.data = "scan_started";
    scan_done_pub_->publish(status);

    acquisition_thread_ = std::thread(&StereoFringeProcess::run_acquisition, this);
}

// ── Acquisition loop ──────────────────────────────────────────────────────────

void StereoFringeProcess::run_acquisition()
{
    RCLCPP_INFO(this->get_logger(), "Starting fringe scan (%zu patterns)…", all_imgs_.size());
    auto t_start = std::chrono::steady_clock::now();

    fringe_process_ptr_->clear_images();
    const size_t total = all_imgs_.size();

    // Timeout per frame: 4× the camera period as generous headroom
    const int camera_hz   = this->get_parameter("camera_hz").as_int();
    const auto frame_timeout = std::chrono::milliseconds(
        std::max(250, 4000 / std::max(camera_hz, 1)));

    for (size_t k = 0; k < total && acquiring_; ++k) {

        // 1. Project pattern k and wait for projector to settle
        cv::imshow(window_name_, all_imgs_[k]);
        cv::waitKey(settle_ms_);

        // 2. Send hardware / software trigger
        send_trigger();

        // 3. Block until one frame arrives on each camera (mirrors triggerAndReceive)
        cv::Mat left  = pop_left(frame_timeout);
        cv::Mat right = pop_right(frame_timeout);

        if (left.empty() || right.empty()) {
            if (!acquiring_) break;
            RCLCPP_WARN(this->get_logger(),
                        "Frame timeout at pattern %zu/%zu – re-triggering…", k + 1, total);
            send_trigger();
            left  = pop_left(frame_timeout);
            right = pop_right(frame_timeout);

            if (left.empty() || right.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                             "Frame still missing after retry at step %zu. Skipping.", k + 1);
                // Insert black frame to keep index alignment (same as stereo_fringe_main)
                cv::Size cam_res = receive_camera_info_ ? camera_resolution_ : project_resolution_;
                left  = cv::Mat::zeros(cam_res, CV_8UC1);
                right = cv::Mat::zeros(cam_res, CV_8UC1);
            }
        }

        // 4. Feed into FringeProcess (stores internal copy)
        fringe_process_ptr_->set_images(left, right, static_cast<int>(k));

        RCLCPP_INFO(this->get_logger(), "  [%2zu/%zu] frame captured", k + 1, total);
    }

    // Return projector to black
    cv::imshow(window_name_, black_img_);
    cv::waitKey(1);

    double acq_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    RCLCPP_INFO(this->get_logger(),
                "Acquisition done in %.3f s (%.1f FPS). Computing phase…",
                acq_s, total / std::max(acq_s, 0.001));

    // 5. Phase computation
    try {
        auto results = fringe_process_ptr_->calculate_abs_phi_images(false);
        publish_processed_images(results);

        auto status = std_msgs::msg::String();
        status.data = "scan_complete";
        scan_done_pub_->publish(status);
        RCLCPP_INFO(this->get_logger(), "Phase processing and publishing complete.");
    } catch (const std::exception& ex) {
        RCLCPP_ERROR(this->get_logger(), "Phase calculation error: %s", ex.what());
        auto status = std_msgs::msg::String();
        status.data = "scan_error";
        scan_done_pub_->publish(status);
    }

    acquiring_ = false;
}

// ── Trigger helper ────────────────────────────────────────────────────────────

void StereoFringeProcess::send_trigger()
{
    if (!trigger_client_->service_is_ready()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Trigger service not ready!");
        return;
    }
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    trigger_client_->async_send_request(req,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
            auto resp = future.get();
            if (!resp->success) {
                RCLCPP_ERROR(this->get_logger(), "Hardware trigger pulse returned failure!");
            }
        });
}

// ── Result publishing ─────────────────────────────────────────────────────────

void StereoFringeProcess::publish_processed_images(const std::vector<cv::Mat>& images)
{
    if (images.size() < 4) {
        RCLCPP_WARN(this->get_logger(),
                    "publish_processed_images: expected 4, got %zu", images.size());
        return;
    }

    auto now = this->get_clock()->now();
    std_msgs::msg::Header hdr_l, hdr_r;
    hdr_l.stamp    = now;
    hdr_r.stamp    = now;
    hdr_l.frame_id = "Active/left_camera_link";
    hdr_r.frame_id = "Active/right_camera_link";

    auto publish_norm = [&](rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                             const cv::Mat& img, const std_msgs::msg::Header& hdr) {
        if (img.empty()) return;
        cv::Mat img8u;
        cv::normalize(img, img8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        pub->publish(*cv_bridge::CvImage(hdr, "mono8", img8u).toImageMsg());
    };

    try {
        // 64FC1 phase maps
        pub_abs_left_->publish(*cv_bridge::CvImage(hdr_l, "64FC1", images[0]).toImageMsg());
        pub_abs_right_->publish(*cv_bridge::CvImage(hdr_r, "64FC1", images[1]).toImageMsg());

        // 8UC1 modulation maps (normalised)
        publish_norm(pub_mod_left_,  images[2], hdr_l);
        publish_norm(pub_mod_right_, images[3], hdr_r);

        bool debug = this->get_parameter("debug").as_bool();
        if (debug) {
            if (this->get_parameter("save_image").as_bool()) {
                fringe_process_ptr_->save_abs_phi_txt(images[0], "left_abs_phi.txt");
                fringe_process_ptr_->save_abs_phi_txt(images[1], "right_abs_phi.txt");
            }
            publish_norm(pub_abs_left_debug_,  images[0], hdr_l);
            publish_norm(pub_abs_right_debug_, images[1], hdr_r);
        }
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge publishing error: %s", e.what());
    }
}

// ── /save_image service ───────────────────────────────────────────────────────

void StereoFringeProcess::save_img_srv_cb(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    const std::string path = this->get_parameter("save_path").as_string();
    if (fringe_process_ptr_->save_images(path)) {
        RCLCPP_INFO(this->get_logger(), "Images saved to %s", path.c_str());
        response->success = true;
        response->message = "Images saved";
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to save images");
        response->success = false;
        response->message = "Save failed";
    }
}

// ── /image_project service ────────────────────────────────────────────────────

void StereoFringeProcess::project_cb(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    if (acquiring_) {
        response->success = false;
        response->message = "Cannot change manual projection during active scan";
        return;
    }

    if (request->data) {
        manual_project_idx_ = (manual_project_idx_ + 1) % static_cast<int>(all_imgs_.size());
        cv::imshow(window_name_, all_imgs_[manual_project_idx_]);
        cv::waitKey(1);
        RCLCPP_INFO(this->get_logger(), "Manual projection pattern: %d / %zu",
                    manual_project_idx_, all_imgs_.size());
    } else {
        manual_project_idx_ = 0;
        cv::imshow(window_name_, black_img_);
        cv::waitKey(1);
        RCLCPP_INFO(this->get_logger(), "Manual projection reset to black");
    }
    response->success = true;
}

} // namespace ros2_active_stereo

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoFringeProcess)