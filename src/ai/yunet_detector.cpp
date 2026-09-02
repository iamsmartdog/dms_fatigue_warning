#include "ai/yunet_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "utils/log.h"

namespace dms {

// 前向声明
static void compute_ear_mar(const std::vector<Point2D>& landmarks, FaceBox& face);
static void estimate_head_pose_pfld(const std::vector<Point2D>& landmarks, uint32_t img_w, uint32_t img_h, float& pitch, float& yaw, float& roll);

// ===== YuNetDetector::Impl =====
struct YuNetDetector::Impl {
    cv::Ptr<cv::FaceDetectorYN> detector;
    std::string model_path;
    float brightness_smooth = 0.0f;  // 平滑后的亮度（EMA 滤波）

    Impl() = default;
    ~Impl() = default;

    bool init(const std::string& path) {
        try {
            // 创建 YuNet 检测器
            // 参数说明：
            // - path: ONNX 模型路径
            // - "" : 配置文件（YuNet 不需要）
            // - Size(320, 320): 模型输入尺寸（固定）
            // - 0.5: 置信度阈值（低光下降低阈值提高召回率）
            // - 0.3: NMS 阈值
            // - 5000: 保留的检测框上限
            detector = cv::FaceDetectorYN::create(
                path,                      // model path
                "",                        // config (empty for ONNX)
                cv::Size(320, 320),        // input size
                0.5f,                      // confidence threshold (降低以提升低光召回)
                0.3f,                      // NMS threshold
                5000,                      // top_k
                cv::dnn::DNN_BACKEND_OPENCV,  // backend
                cv::dnn::DNN_TARGET_CPU    // target
            );

            if (!detector) {
                LOGE("YUNET", "Failed to create detector from: %s", path.c_str());
                return false;
            }

            model_path = path;
            LOGI("YUNET", "Detector created successfully from: %s", path.c_str());
            return true;
        } catch (const std::exception& e) {
            LOGE("YUNET", "Exception during init: %s", e.what());
            return false;
        }
    }

    bool detect(const cv::Mat& bgr_img, std::vector<cv::Mat>& faces) {
        if (!detector) {
            LOGE("YUNET", "Detector not initialized");
            return false;
        }

        try {
            // 设置输入尺寸为图像实际尺寸
            detector->setInputSize(bgr_img.size());

            // 执行检测
            cv::Mat result;
            detector->detect(bgr_img, result);

            // 检测结果存储在 faces 中（调用方负责使用）
            if (!result.empty()) {
                faces.clear();
                faces.push_back(result);
            }

            return true;
        } catch (const std::exception& e) {
            LOGE("YUNET", "Exception during detect: %s", e.what());
            return false;
        }
    }
};

// ===== YuNetDetector 实现 =====

YuNetDetector::YuNetDetector(const std::string& model_path,
                               const std::string& pfld_path)
    : model_path_(model_path), pfld_model_path_(pfld_path),
      impl_(std::make_unique<Impl>()) {
    if (!pfld_path.empty()) {
        pfld_ = std::make_unique<PFLDDetector>(pfld_path);
    }
}

YuNetDetector::~YuNetDetector() = default;

bool YuNetDetector::init() {
    if (model_path_.empty()) {
        LOGE("YUNET", "Model path not set");
        return false;
    }

    if (!impl_->init(model_path_)) {
        return false;
    }

    // 初始化 PFLD 关键点检测器（可选）
    if (pfld_) {
        if (!pfld_->init()) {
            LOGE("YUNET", "PFLD init failed, EAR/MAR will not be available");
            pfld_.reset();  // 禁用 PFLD，仍可用人脸检测
        }
    }

    return true;
}

bool YuNetDetector::detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                            DetectionResult& result) {
    if (!rgb || w == 0 || h == 0) {
        LOGE("YUNET", "Invalid input: null pointer or zero size");
        return false;
    }

    try {
        // RGB -> BGR 转换（OpenCV 默认使用 BGR）
        cv::Mat rgb_img(h, w, CV_8UC3, (void*)rgb);
        cv::Mat bgr_img;
        cv::cvtColor(rgb_img, bgr_img, cv::COLOR_RGB2BGR);

        // 应用低光增强
        // 根据亮度判断是否需要增强
        cv::Mat gray;
        cv::cvtColor(bgr_img, gray, cv::COLOR_BGR2GRAY);
        double avg_brightness = cv::mean(gray)[0];

        // EMA 平滑亮度，避免闪烁
        if (impl_->brightness_smooth == 0.0f) {
            impl_->brightness_smooth = static_cast<float>(avg_brightness);
        } else {
            impl_->brightness_smooth =
                0.9f * impl_->brightness_smooth + 0.1f * static_cast<float>(avg_brightness);
        }

        cv::Mat input_img = bgr_img.clone();
        // 滞回阈值：进入夜间 < 100，退出夜间 > 130，避免在阈值附近来回切换
        if (impl_->brightness_smooth < 100.0f) {
            // === 极低光增强策略 ===
            // 1. 先降噪（避免噪声被放大）——用高斯模糊，避免依赖 photo 模块
            cv::Mat denoised;
            cv::GaussianBlur(gray, denoised, cv::Size(3, 3), 0);

            // 2. 线性拉伸对比度（直接拉满）
            cv::Mat stretched;
            cv::normalize(denoised, stretched, 0, 255, cv::NORM_MINMAX);

            // 3. CLAHE 增强（比之前更强）
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(4, 4));
            clahe->apply(stretched, stretched);

            // 4. Gamma 校正（大幅提亮暗部）
            cv::Mat gamma_enhanced;
            stretched.convertTo(gamma_enhanced, CV_32F, 1.0/255.0);
            cv::pow(gamma_enhanced, 0.5, gamma_enhanced);  // gamma=0.5 大幅提亮
            gamma_enhanced.convertTo(gray, CV_8U, 255.0);

            // 转回 BGR 用于检测
            cv::cvtColor(gray, input_img, cv::COLOR_GRAY2BGR);
        }

        // 执行检测
        std::vector<cv::Mat> faces;
        if (!impl_->detect(input_img, faces)) {
            LOGE("YUNET", "Detection failed");
            return false;
        }

        // 解析检测结果
        result.faces.clear();
        if (!faces.empty() && !faces[0].empty()) {
            const cv::Mat& detections = faces[0];

            // YuNet 输出格式（每一行是一个检测）：
            // [0-1]: 人脸框左上角 (x, y)
            // [2-3]: 人脸框宽高 (w, h)
            // [4-13]: 5 个关键点坐标 (2*5=10 个值)
            //   - [4-5]: 右眼
            //   - [6-7]: 左眼
            //   - [8-9]: 鼻子
            //   - [10-11]: 右嘴角
            //   - [12-13]: 左嘴角
            // [14]: 置信度分数

            for (int i = 0; i < detections.rows; ++i) {
                float conf = detections.at<float>(i, 14);

                // 置信度过低则跳过（低光下放宽到 0.4）
                if (conf < 0.4f) {
                    continue;
                }

                FaceBox face;
                face.x = static_cast<float>(detections.at<float>(i, 0));
                face.y = static_cast<float>(detections.at<float>(i, 1));
                face.w = static_cast<float>(detections.at<float>(i, 2));
                face.h = static_cast<float>(detections.at<float>(i, 3));
                face.score = conf;

                // 提取 5 个关键点
                // YuNet 输出顺序：右眼、左眼、鼻子、右嘴角、左嘴角
                face.landmarks.resize(5);
                for (int j = 0; j < 5; ++j) {
                    face.landmarks[j].x =
                        static_cast<float>(detections.at<float>(i, 4 + 2 * j));
                    face.landmarks[j].y =
                        static_cast<float>(detections.at<float>(i, 5 + 2 * j));
                }

                // 使用 PFLD 106 点精确计算 EAR/MAR（如果可用）
                if (pfld_) {
                    std::vector<Point2D> lm106;
                    if (pfld_->detect(rgb, w, h, face, lm106) &&
                        lm106.size() >= 106) {
                        // PFLD 106 点索引：
                        // [40-59] 左眼, [60-79] 右眼, [88-95] 左嘴角, [96-103] 右嘴角
                        compute_ear_mar(lm106, face);
                        // 用 106 点更新姿态估计（YuNet 5 点不够）
                         estimate_head_pose_pfld(lm106, w, h,
                                                 face.head_pitch, face.head_yaw,
                                                 face.head_roll);
                    }
                    // 调试：始终打印 EAR 值
                    static int dbg_cnt = 0;
                    if (++dbg_cnt % 30 == 0) {
                        LOGI("YUNET", "[DBG] PFLD face=(%.0f,%.0f %.0fx%.0f) "
                             "earL=%.3f earR=%.3f mar=%.3f",
                             face.x, face.y, face.w, face.h,
                             face.ear_left, face.ear_right, face.mar);
                    }
                }

                result.faces.push_back(face);
            }
        }

        return true;
    } catch (const std::exception& e) {
        LOGE("YUNET", "Exception in detect: %s", e.what());
        return false;
    }
}


// ===== Helper Functions =====

// 计算眼睛长宽比 (EAR: Eye Aspect Ratio)
static float compute_eye_aspect_ratio(const std::vector<Point2D>& landmarks,
                                       int p1, int p2, int p3, int p4,
                                       int p5, int p6) {
    if (p1 < 0 || p2 < 0 || p3 < 0 || p4 < 0 || p5 < 0 || p6 < 0 ||
        p1 >= static_cast<int>(landmarks.size()) ||
        p2 >= static_cast<int>(landmarks.size()) ||
        p3 >= static_cast<int>(landmarks.size()) ||
        p4 >= static_cast<int>(landmarks.size()) ||
        p5 >= static_cast<int>(landmarks.size()) ||
        p6 >= static_cast<int>(landmarks.size())) {
        return 0.0f;
    }

    float dist_vertical1 =
        std::sqrt(std::pow(landmarks[p2].x - landmarks[p5].x, 2) +
                  std::pow(landmarks[p2].y - landmarks[p5].y, 2));
    float dist_vertical2 =
        std::sqrt(std::pow(landmarks[p3].x - landmarks[p4].x, 2) +
                  std::pow(landmarks[p3].y - landmarks[p4].y, 2));
    float dist_horizontal =
        std::sqrt(std::pow(landmarks[p1].x - landmarks[p6].x, 2) +
                  std::pow(landmarks[p1].y - landmarks[p6].y, 2));

    if (dist_horizontal < 1e-6f) return 0.0f;
    return (dist_vertical1 + dist_vertical2) / (2.0f * dist_horizontal);
}

// 计算嘴巴长宽比 (MAR: Mouth Aspect Ratio)
static float compute_mouth_aspect_ratio(const std::vector<Point2D>& landmarks,
                                        int top, int bottom, int left,
                                        int right) {
    if (top < 0 || bottom < 0 || left < 0 || right < 0 ||
        top >= static_cast<int>(landmarks.size()) ||
        bottom >= static_cast<int>(landmarks.size()) ||
        left >= static_cast<int>(landmarks.size()) ||
        right >= static_cast<int>(landmarks.size())) {
        return 0.0f;
    }

    float dist_vertical =
        std::sqrt(std::pow(landmarks[top].x - landmarks[bottom].x, 2) +
                  std::pow(landmarks[top].y - landmarks[bottom].y, 2));
    float dist_horizontal =
        std::sqrt(std::pow(landmarks[left].x - landmarks[right].x, 2) +
                  std::pow(landmarks[left].y - landmarks[right].y, 2));

    if (dist_horizontal < 1e-6f) return 0.0f;
    return dist_vertical / dist_horizontal;
}

// 计算 EAR 和 MAR
static void compute_ear_mar(const std::vector<Point2D>& landmarks,
                             FaceBox& face) {
    // PFLD 106 点索引：
    // [40-59]: 左眼 (20 个点)
    // [60-79]: 右眼 (20 个点)
    // [88-95]: 左嘴角 (8 个点)
    // [96-103]: 右嘴角 (8 个点)

    if (landmarks.size() < 104) {
        face.ear_left = 0.0f;
        face.ear_right = 0.0f;
        face.mar = 0.0f;
        return;
    }

    // 计算左眼 EAR
    float left_ear = compute_eye_aspect_ratio(landmarks, 40, 43, 47, 46, 44, 50);

    // 计算右眼 EAR
    float right_ear = compute_eye_aspect_ratio(landmarks, 60, 63, 67, 66, 64, 70);

    // 赋值到 FaceBox
    face.ear_left = left_ear;
    face.ear_right = right_ear;

    // 计算嘴巴长宽比（使用左右嘴角的关键点）
    face.mar = compute_mouth_aspect_ratio(landmarks, 88, 92, 90, 94);
}

// 估计头部姿态（基于 106 个关键点）
static void estimate_head_pose_pfld(const std::vector<Point2D>& landmarks,
                                    uint32_t img_w, uint32_t img_h, float& pitch,
                                    float& yaw, float& roll) {
    if (landmarks.size() < 106) {
        pitch = 0.0f;
        yaw = 0.0f;
        roll = 0.0f;
        return;
    }

    // 使用眼睛和鼻子的相对位置估计头部姿态
    // 简单实现：基于特定关键点的几何关系
    float left_eye_x = landmarks[40].x;
    float left_eye_y = landmarks[40].y;
    float right_eye_x = landmarks[60].x;
    float right_eye_y = landmarks[60].y;
    float nose_x = landmarks[76].x;  // 鼻子（假设索引）
    float nose_y = landmarks[76].y;

    // Pitch: 鼻子相对于眼睛的垂直偏移
    float eye_center_y = (left_eye_y + right_eye_y) / 2.0f;
    pitch = (nose_y - eye_center_y) / img_h * 45.0f;  // 归一化到 [-45, 45]

    // Yaw: 鼻子相对于眼睛的水平偏移
    float eye_center_x = (left_eye_x + right_eye_x) / 2.0f;
    yaw = (nose_x - eye_center_x) / img_w * 45.0f;  // 归一化到 [-45, 45]

    // Roll: 两眼连线的倾斜角
    float eye_dx = right_eye_x - left_eye_x;
    float eye_dy = right_eye_y - left_eye_y;
    roll = std::atan2(eye_dy, eye_dx) * 180.0f / 3.14159265f;
}

}  // namespace dms
