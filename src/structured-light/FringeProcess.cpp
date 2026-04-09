#include "FringeProcess.hpp"
#include <chrono>

FringeProcess::FringeProcess(cv::Size img_res, cv::Size cam_res, int px_f, int steps)
    : FringePattern(img_res, px_f, steps), GrayCode(img_res, 0, px_f),
      cam_width(cam_res.width), cam_height(cam_res.height) {

    n_bits = get_gc_images("gray").size(); // n_bits + 2
    total_steps = steps + n_bits;

    images_left.resize(total_steps);
    images_right.resize(total_steps);

    for (int i = 0; i < total_steps; ++i) {
        images_left[i] = cv::Mat::zeros(cam_height, cam_width, CV_8UC1);
        images_right[i] = cv::Mat::zeros(cam_height, cam_width, CV_8UC1);
    }
}

bool FringeProcess::save_images(std::string path){
    try {
        // Cria os diretórios recursivamente como um mkdir -p
        std::filesystem::create_directories(path + "/left");
        std::filesystem::create_directories(path + "/right");
    } catch (const std::filesystem::filesystem_error& e) {
        return false;
    }

    for (size_t i = 0; i < images_left.size(); ++i) {
        // Formata o contador com preenchimento de zeros (000, 001, etc)
        std::stringstream ss;
        ss << std::setw(3) << std::setfill('0') << i;
        std::string img_counter = ss.str();

        std::string path_L = path + "/left/L" + img_counter + ".png";
        std::string path_R = path + "/right/R" + img_counter + ".png";

        // Salva as matrizes OpenCV e valida o sucesso da gravação
        if (!cv::imwrite(path_L, images_left[i]) || !cv::imwrite(path_R, images_right[i])) {
            return false;
        }
    }
    return true;
}

int FringeProcess::get_total_steps(){
    return total_steps;
}

void FringeProcess::set_images(const cv::Mat& left, const cv::Mat& right, int counter) {
    if (counter < total_steps) {
        left.copyTo(images_left[counter]);
        right.copyTo(images_right[counter]);
    }
}

void FringeProcess::set_camera_resolution(cv::Size cam_resolution){
    cam_width = cam_resolution.width;
    cam_height = cam_resolution.height;
    int steps = get_steps();
    total_steps = steps + n_bits;
    images_left.clear();
    images_right.clear();
    images_left.resize(total_steps);
    images_right.resize(total_steps);

    for (int i = 0; i < total_steps; ++i) {
        images_left[i] = cv::Mat::zeros(cam_height, cam_width, CV_8UC1);
        images_right[i] = cv::Mat::zeros(cam_height, cam_width, CV_8UC1);
    }

}

std::pair<cv::Mat, cv::Mat> FringeProcess::calculate_phi(const std::vector<cv::Mat>& images) {
    int h = images[0].rows;
    int w = images[0].cols;
    int N = images.size();

    cv::Mat phi_image(h, w, CV_64FC1);
    cv::Mat modulation_map(h, w, CV_64FC1);

    std::vector<double> sin_lut(N), cos_lut(N);
    for (int i = 0; i < N; ++i) {
        double angle = 2.0 * CV_PI * (i + 1.0) / N;
        sin_lut[i] = sin(angle);
        cos_lut[i] = cos(angle);
    }

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double sin_sum = 0, cos_sum = 0;
            for (int k = 0; k < N; ++k) {
                double val = static_cast<double>(images[k].at<uchar>(i, j));
                sin_sum += val * sin_lut[k];
                cos_sum += val * cos_lut[k];
            }
            phi_image.at<double>(i, j) = atan2(-sin_sum, cos_sum);
            // std::cout << atan2(-sin_sum, cos_sum) << std::endl;
            modulation_map.at<double>(i, j) = sqrt(sin_sum * sin_sum + cos_sum * cos_sum)/N;
            // std::cout << sqrt(sin_sum * sin_sum + cos_sum * cos_sum) << std::endl;
        }
    }
    return {modulation_map, phi_image};
}

cv::Mat FringeProcess::calculate_qsi(const std::vector<cv::Mat>& gc_imgs) {
    int h = gc_imgs[0].rows;
    int w = gc_imgs[0].cols;
    int bits_count = gc_imgs.size() - 2;
    cv::Mat qsi_image = cv::Mat::zeros(h, w, CV_32SC1);

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            double white = std::max(static_cast<double>(gc_imgs[0].at<uchar>(i, j)), 1e-6);
            int val = 0;
            for (int k = 0; k < bits_count; ++k) {
                if ((static_cast<double>(gc_imgs[k + 2].at<uchar>(i, j)) / white) > 0.5) {
                    val |= (1 << (bits_count - 1 - k));
                }
            }
            qsi_image.at<int>(i, j) = val;
        }
    }
    return qsi_image;
}

cv::Mat FringeProcess::remap_qsi_image(const cv::Mat& qsi_image, const std::vector<int>& real_order) {
    std::map<int, int> mapping;
    for (int i = 0; i < real_order.size(); ++i) mapping[real_order[i]] = i;

    cv::Mat remapped = cv::Mat::zeros(qsi_image.size(), CV_32SC1);
    for (int i = 0; i < qsi_image.rows; ++i) {
        for (int j = 0; j < qsi_image.cols; ++j) {
            int val = qsi_image.at<int>(i, j);
            remapped.at<int>(i, j) = mapping.count(val) ? mapping[val] : 0;
        }
    }
    return remapped;
}

std::vector<cv::Mat> FringeProcess::calculate_abs_phi_images() {
    auto start = std::chrono::steady_clock::now();

    // Separando sub-vetores (Simulando o slicing do Python)
    std::vector<cv::Mat> gc_l(images_left.begin(), images_left.begin() + n_bits);
    std::vector<cv::Mat> fr_l(images_left.begin() + n_bits, images_left.end());
    
    std::vector<cv::Mat> gc_r(images_right.begin(), images_right.begin() + n_bits);
    std::vector<cv::Mat> fr_r(images_right.begin() + n_bits, images_right.end());

    auto [mod_l, phi_l] = calculate_phi(fr_l);
    auto [mod_r, phi_r] = calculate_phi(fr_r);
    

    cv::Mat remap_r = remap_qsi_image(calculate_qsi(gc_r),get_gc_order_v());
    cv::Mat remap_l = remap_qsi_image(calculate_qsi(gc_l), get_gc_order_v());

    cv::Mat phi_remap;
    phi_l.convertTo(phi_remap, CV_8U, 255.0 / (2.0 * CV_PI * 20));
    cv::imshow("Phi R", phi_remap);
    cv::waitKey(0);
    cv::imshow("Phi L", phi_l);
    cv::waitKey(0);
    cv::Mat r_l;
    remap_l.convertTo(r_l, CV_8U, 255.0 / (20));
    cv::imshow("remap_l", r_l);
    cv::waitKey(0);


    auto compute_abs = [&](const cv::Mat& phi, const cv::Mat& qsi) {
        cv::Mat abs_phi = cv::Mat::zeros(phi.size(), CV_64FC1);
        for(int i=0; i<phi.rows; ++i) {
            for(int j=0; j<phi.cols; ++j) {
                double p = phi.at<double>(i, j);
                int q = qsi.at<int>(i, j);
                double k;
                if (p <= -CV_PI/2.0) k = std::floor((q + 1) / 2.0);
                else if (p < CV_PI/2.0) k = std::floor(q / 2.0);
                else k = std::floor((q + 1) / 2.0) - 1.0;
                
                abs_phi.at<double>(i, j) = p + 2.0 * CV_PI * k + CV_PI;
            }
        }
        return abs_phi;
    };

    cv::Mat abs_l = compute_abs(phi_l, remap_l);
    cv::Mat abs_r = compute_abs(phi_r, remap_r);

    auto end = std::chrono::steady_clock::now();
    std::cout << "Process abs phase: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

    return {abs_l, abs_r, mod_l, mod_r};
}