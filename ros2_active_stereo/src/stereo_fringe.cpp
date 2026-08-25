#include <stereo_fringe.hpp>
#include <chrono>
#include <mutex>

namespace ros2_active_stereo
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
StereoFringeProcess::StereoFringeProcess(const rclcpp::NodeOptions & options)
: Node("image_project_node", options)
{
    // ── Parameters ────────────────────────────────────────────────────────
    this->declare_parameter("monitor_name",      "Monitor_1");
    this->declare_parameter("pixel_per_fringe",  128);
    this->declare_parameter("fringe_steps",      4);
    this->declare_parameter("image_color",       "blue");
    this->declare_parameter("camera_hz",         20);
    this->declare_parameter("save_path",         "/tmp/structured-light");
    this->declare_parameter("save_image",        true);
    this->declare_parameter("debug",             true);
    // Settling delay in ms: time between imshow() and sending the camera trigger.
    // At 60 Hz the projector refreshes every ~16.7 ms; add a small margin.
    // Default 22 ms is safe for most configurations — tune down if needed.
    this->declare_parameter("settle_ms",         22);

    pixel_per_fringe = this->get_parameter("pixel_per_fringe").as_int();
    fringe_steps     = this->get_parameter("fringe_steps").as_int();
    color_           = this->get_parameter("image_color").as_string();
    // Display timer runs at ≤ projector frame rate (keep OpenCV window alive)
    // as_int() returns int64_t; cast to int before arithmetic to avoid type mismatch.
    int camera_hz = static_cast<int>(this->get_parameter("camera_hz").as_int());
    timer_hz_ = 1000.0 / std::max(camera_hz, 1);

    // ── Screen & window ───────────────────────────────────────────────────
    if (!get_screen_resolution(this->get_parameter("monitor_name").as_string())) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get screen resolution");
        return;
    }
    construct_window();

    // ── Pattern generation ────────────────────────────────────────────────
    fringe_process_ptr_ = std::make_unique<FringeProcess>(
        project_resolution_,
        cv::Size(2448, 2048),
        pixel_per_fringe,
        fringe_steps);

    fringe_process_ptr_->create_fringe_image();
    fringe_process_ptr_->create_graycode_image();

    all_imgs_.clear();
    all_imgs_.push_back(black_img_);   // index 0: black warm-up frame (never stored)
    auto gc_imgs = fringe_process_ptr_->get_gc_images(color_);
    auto fr_imgs = fringe_process_ptr_->get_fr_images(color_);
    all_imgs_.insert(all_imgs_.end(), gc_imgs.begin(), gc_imgs.end());
    all_imgs_.insert(all_imgs_.end(), fr_imgs.begin(), fr_imgs.end());

    RCLCPP_INFO(this->get_logger(), "Total patterns (incl. black): %zu", all_imgs_.size());

    // ── Callback groups ───────────────────────────────────────────────────
    // Mutually-exclusive groups ensure thread safety without manual locks
    // for calls within the same group.
    display_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    srv_cb_group_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    stereo_cb_group_  = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    // ── Subscribers ───────────────────────────────────────────────────────
    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(2);
    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = stereo_cb_group_;

    sub_left_.subscribe(this, "left/image_raw",  qos.get_rmw_qos_profile(), sub_options);
    sub_right_.subscribe(this, "right/image_raw", qos.get_rmw_qos_profile(), sub_options);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                SyncPolicy(15), sub_left_, sub_right_);
    sync_->registerCallback(
        std::bind(&StereoFringeProcess::stereo_callback, this,
                  std::placeholders::_1, std::placeholders::_2));

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

    // Status topic: subscribers can listen for "scan_started" / "scan_complete" / "scan_error"
    scan_done_pub_ = this->create_publisher<std_msgs::msg::String>("fringe_status", 10);

    // ── Services / clients ────────────────────────────────────────────────
    auto srv_qos = rclcpp::ServicesQoS();
    change_image_service_ = this->create_service<std_srvs::srv::SetBool>(
        "image_project",
        std::bind(&StereoFringeProcess::project_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        srv_qos, srv_cb_group_);

    process_service_ = this->create_service<std_srvs::srv::Trigger>(
        "phase_process",
        std::bind(&StereoFringeProcess::process_srv_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        srv_qos, srv_cb_group_);

    save_imgs_service_ = this->create_service<std_srvs::srv::Trigger>(
        "save_image",
        std::bind(&StereoFringeProcess::save_img_srv_cb, this,
                  std::placeholders::_1, std::placeholders::_2),
        srv_qos, srv_cb_group_);

    trigger_client_ = this->create_client<std_srvs::srv::Trigger>(
        "trigger", srv_qos, srv_cb_group_);

    // ── Display timer (keeps OpenCV window alive; does NOT drive acquisition)
    display_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<long>(timer_hz_)),
        std::bind(&StereoFringeProcess::display_timer_cb, this),
        display_cb_group_);
}

// ─────────────────────────────────────────────────────────────────────────────
StereoFringeProcess::~StereoFringeProcess() {
    cv::destroyWindow(window_name_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Screen / window helpers
// ─────────────────────────────────────────────────────────────────────────────
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
            RCLCPP_INFO(this->get_logger(), "Selected monitor '%s'", monitor_name.c_str());
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
    // Show black at startup so the projector is dark until a scan begins
    cv::imshow(window_name_, black_img_);
    cv::waitKey(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Display timer — keeps the window responsive; no acquisition logic here
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::display_timer_cb()
{
    if (!receive_camera_info_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Waiting for camera_info…");
    }
    // Pump the OpenCV event loop so the window stays alive and the OS
    // compositor receives the latest imshow() framebuffer.
    cv::waitKey(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera info callback
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::camera_info_cb(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
    if (receive_camera_info_) return;
    fringe_process_ptr_->set_camera_resolution(cv::Size(msg->width, msg->height));
    RCLCPP_INFO(this->get_logger(), "Camera resolution: %ux%u", msg->width, msg->height);
    receive_camera_info_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// process_srv_cb — starts the scan; returns immediately so callers don't block.
// Listen to /fringe_status for "scan_complete" or "scan_error".
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::process_srv_cb(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    {
        std::lock_guard<std::mutex> lk(scan_mtx_);
        if (scan_state_ != ScanState::IDLE) {
            response->success = false;
            response->message = "Scan already in progress";
            return;
        }
        if (!receive_camera_info_) {
            response->success = false;
            response->message = "Camera info not yet received";
            return;
        }
        if (!trigger_client_->service_is_ready()) {
            response->success = false;
            response->message = "Trigger service not ready";
            return;
        }

        // Cache parameters for this scan so we don't call get_parameter() in callbacks
        cached_settle_ms_ = this->get_parameter("settle_ms").as_int();
        cached_debug_     = this->get_parameter("debug").as_bool();
        cached_color_     = this->get_parameter("image_color").as_string();

        // Check if pattern parameters changed and rebuild if needed
        int px_f  = this->get_parameter("pixel_per_fringe").as_int();
        int steps = this->get_parameter("fringe_steps").as_int();
        if (px_f != pixel_per_fringe || steps != fringe_steps) {
            RCLCPP_INFO(this->get_logger(),
                        "Rebuilding patterns: px_f=%d steps=%d", px_f, steps);
            pixel_per_fringe = px_f;
            fringe_steps     = steps;
            fringe_process_ptr_->FringePattern::set_px_f(pixel_per_fringe);
            fringe_process_ptr_->FringePattern::set_steps(fringe_steps);
            fringe_process_ptr_->GrayCode::set_px_f(pixel_per_fringe);
            fringe_process_ptr_->create_fringe_image();
            fringe_process_ptr_->create_graycode_image();

            auto gc_imgs = fringe_process_ptr_->get_gc_images(cached_color_);
            auto fr_imgs = fringe_process_ptr_->get_fr_images(cached_color_);
            all_imgs_.clear();
            all_imgs_.push_back(black_img_);
            all_imgs_.insert(all_imgs_.end(), gc_imgs.begin(), gc_imgs.end());
            all_imgs_.insert(all_imgs_.end(), fr_imgs.begin(), fr_imgs.end());
        }

        fringe_process_ptr_->clear_images();
        scan_index_ = 0;
        scan_state_ = ScanState::SETTLING;
    }

    // Publish black during warm-up, then kick off first pattern
    RCLCPP_INFO(this->get_logger(),
                "Scan started: %zu patterns, settle=%d ms",
                all_imgs_.size() - 1, cached_settle_ms_);

    auto status = std_msgs::msg::String();
    status.data = "scan_started";
    scan_done_pub_->publish(status);

    response->success = true;
    response->message = "Scan started; listen to /fringe_status for completion";

    // Kick off the state machine — project pattern 0 (black warm-up)
    advance_scan_step();
}

// ─────────────────────────────────────────────────────────────────────────────
// advance_scan_step — project current pattern, arm settling one-shot timer
// Must be called with scan_mtx_ NOT held (creates a timer internally).
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::advance_scan_step()
{
    // Project pattern at scan_index_
    {
        std::lock_guard<std::mutex> lk(scan_mtx_);
        cv::imshow(window_name_, all_imgs_[scan_index_]);
        // cv::waitKey(1) is called by the display timer, which runs concurrently.
        // Call it here too so the OS compositor actually flips the buffer before
        // the settling delay starts.
        cv::waitKey(1);
    }

    // Arm one-shot settling timer (cancels previous if any)
    if (settling_timer_) {
        settling_timer_->cancel();
    }
    settling_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(cached_settle_ms_),
        [this]() {
            settling_timer_->cancel(); // one-shot: self-cancel
            settling_done_cb();
        },
        srv_cb_group_);
}

// ─────────────────────────────────────────────────────────────────────────────
// settling_done_cb — projector has had time to display the pattern; trigger now
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::settling_done_cb()
{
    {
        std::lock_guard<std::mutex> lk(scan_mtx_);
        scan_state_ = ScanState::WAITING_FOR_FRAME;
    }
    send_trigger();
}

// ─────────────────────────────────────────────────────────────────────────────
// stereo_callback — image pair arrived after hardware trigger
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::stereo_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
{
    ScanState current_state;
    int current_index;
    {
        std::lock_guard<std::mutex> lk(scan_mtx_);
        current_state = scan_state_;
        current_index = scan_index_;
    }

    if (current_state != ScanState::WAITING_FOR_FRAME) {
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                              "stereo_callback: not waiting for frame (state=%d), ignoring",
                              static_cast<int>(current_state));
        return;
    }

    try {
        cv::Mat left  = cv_bridge::toCvShare(left_msg,  "mono8")->image;
        cv::Mat right = cv_bridge::toCvShare(right_msg, "mono8")->image;

        if (left.empty() || right.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty frame received at index %d", current_index);
            return;
        }

        // index 0 is the black warm-up frame — skip saving it
        if (current_index > 0) {
            fringe_process_ptr_->set_images(left, right, current_index - 1);
            RCLCPP_DEBUG(this->get_logger(),
                         "Stored frame %d/%zu", current_index, all_imgs_.size() - 1);
        }

        // Advance to next pattern
        int next_index;
        {
            std::lock_guard<std::mutex> lk(scan_mtx_);
            scan_index_++;
            next_index = scan_index_;

            if (next_index >= static_cast<int>(all_imgs_.size())) {
                // All patterns captured → transition to PROCESSING
                scan_state_ = ScanState::PROCESSING;
            } else {
                scan_state_ = ScanState::SETTLING;
            }
        }

        if (next_index >= static_cast<int>(all_imgs_.size())) {
            // ── Processing ─────────────────────────────────────────────────
            RCLCPP_INFO(this->get_logger(),
                        "All %zu patterns captured. Computing phase maps…",
                        all_imgs_.size() - 1);

            // Show black while processing
            cv::imshow(window_name_, black_img_);
            cv::waitKey(1);

            std::vector<cv::Mat> result;
            try {
                result = fringe_process_ptr_->calculate_abs_phi_images(false);
            } catch (const std::exception& ex) {
                RCLCPP_ERROR(this->get_logger(), "Phase computation failed: %s", ex.what());
                auto status = std_msgs::msg::String();
                status.data = "scan_error";
                scan_done_pub_->publish(status);
                std::lock_guard<std::mutex> lk(scan_mtx_);
                scan_state_ = ScanState::IDLE;
                return;
            }

            publish_processed_images(result);

            auto status = std_msgs::msg::String();
            status.data = "scan_complete";
            scan_done_pub_->publish(status);
            RCLCPP_INFO(this->get_logger(), "Scan complete — phase maps published.");

            std::lock_guard<std::mutex> lk(scan_mtx_);
            scan_state_ = ScanState::IDLE;
        } else {
            // ── Next pattern ────────────────────────────────────────────────
            RCLCPP_DEBUG(this->get_logger(),
                         "Pattern %d/%zu captured, projecting next…",
                         current_index, all_imgs_.size() - 1);
            advance_scan_step();
        }

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// send_trigger — async, zero-blocking
// ─────────────────────────────────────────────────────────────────────────────
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
                RCLCPP_ERROR(this->get_logger(), "Hardware trigger failed: %s",
                             resp->message.c_str());
            }
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// publish_processed_images — {abs_phi_l, abs_phi_r, mod_l, mod_r}
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::publish_processed_images(const std::vector<cv::Mat>& images)
{
    if (images.size() < 4) {
        RCLCPP_WARN(this->get_logger(),
                    "publish_processed_images: expected 4 images, got %zu", images.size());
        return;
    }

    auto now = this->get_clock()->now();
    std_msgs::msg::Header hdr_l, hdr_r;
    hdr_l.stamp    = now;
    hdr_r.stamp    = now;
    hdr_l.frame_id = "Active/left_camera_link";
    hdr_r.frame_id = "Active/right_camera_link";

    // Helper: normalize float mat to 8-bit and publish
    auto publish_norm = [&](rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                             const cv::Mat& img, const std_msgs::msg::Header& hdr) {
        if (img.empty()) return;
        cv::Mat img8u;
        cv::normalize(img, img8u, 0, 255, cv::NORM_MINMAX, CV_8U);
        pub->publish(*cv_bridge::CvImage(hdr, "mono8", img8u).toImageMsg());
    };

    try {
        // 64FC1 phase maps
        pub_abs_left_->publish(
            *cv_bridge::CvImage(hdr_l, "64FC1", images[0]).toImageMsg());
        pub_abs_right_->publish(
            *cv_bridge::CvImage(hdr_r, "64FC1", images[1]).toImageMsg());

        // 8UC1 modulation maps
        publish_norm(pub_mod_left_,  images[2], hdr_l);
        publish_norm(pub_mod_right_, images[3], hdr_r);

        if (cached_debug_) {
            // Save absolute phi as text for offline inspection
            if (this->get_parameter("save_image").as_bool()) {
                fringe_process_ptr_->save_abs_phi_txt(images[0], "left_abs_phi.txt");
                fringe_process_ptr_->save_abs_phi_txt(images[1], "right_abs_phi.txt");
            }
            // Normalised debug views
            publish_norm(pub_abs_left_debug_,  images[0], hdr_l);
            publish_norm(pub_abs_right_debug_, images[1], hdr_r);
        }
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge in publish: %s", e.what());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// save_img_srv_cb
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// project_cb — manual projection control for alignment / testing
// ─────────────────────────────────────────────────────────────────────────────
void StereoFringeProcess::project_cb(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    const std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    std::lock_guard<std::mutex> lk(scan_mtx_);
    if (scan_state_ != ScanState::IDLE) {
        response->success = false;
        response->message = "Cannot change projection during an active scan";
        return;
    }

    if (request->data) {
        scan_index_ = (scan_index_ + 1) % static_cast<int>(all_imgs_.size());
        cv::imshow(window_name_, all_imgs_[scan_index_]);
        cv::waitKey(1);
        RCLCPP_INFO(this->get_logger(), "Manual projection: index %d/%zu",
                    scan_index_, all_imgs_.size() - 1);
    } else {
        scan_index_ = 0;
        cv::imshow(window_name_, black_img_);
        cv::waitKey(1);
        RCLCPP_INFO(this->get_logger(), "Manual projection: black (reset)");
    }
    response->success = true;
}

} // namespace ros2_active_stereo

RCLCPP_COMPONENTS_REGISTER_NODE(ros2_active_stereo::StereoFringeProcess)