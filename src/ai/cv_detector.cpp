#include "ai/cv_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef DMS_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#endif

#include "utils/log.h"

namespace dms {

#ifdef DMS_HAS_OPENCV

struct CvDetector::Impl {
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eye_cascade;
    bool ready = false;
};

CvDetector::CvDetector(const std::string& model_dir)
    : model_dir_(model_dir), impl_(std::make_unique<Impl>()) {}

CvDetector::~CvDetector() = default;

// 在候选目录里查找 haarcascade xml
std::string CvDetector::resolve_xml(const std::string& basename) const {
    std::vector<std::string> dirs;
    if (!model_dir_.empty()) dirs.push_back(model_dir_);
    // apt 安装的 opencv-data
    dirs.push_back("/usr/share/opencv4/haarcascades/");
    dirs.push_back("/usr/share/opencv/haarcascades/");
    // 源码编译安装
    dirs.push_back("/usr/local/share/opencv4/haarcascades/");
    // 源码树（本机虚拟机环境）
    dirs.push_back("/usr/local/src/opencv-4.12.0/data/haarcascades/");
    // 项目自带（部署到板子时使用）
    dirs.push_back("./models/");
    dirs.push_back("/opt/dms/models/");

    for (const auto& d : dirs) {
        std::string p = d + basename;
        std::ifstream ifs(p);
        if (ifs.good()) {
            LOGI("CVDET", "found %s at %s", basename.c_str(), p.c_str());
            return p;
        }
    }
    return "";
}

bool CvDetector::init() {
    std::string face_xml = resolve_xml("haarcascade_frontalface_alt2.xml");
    std::string eye_xml  = resolve_xml("haarcascade_eye.xml");
    if (face_xml.empty() || eye_xml.empty()) {
        LOGE("CVDET", "haarcascade xml not found (face=%s eye=%s)",
             face_xml.c_str(), eye_xml.c_str());
        return false;
    }
    if (!impl_->face_cascade.load(face_xml) ||
        !impl_->eye_cascade.load(eye_xml)) {
        LOGE("CVDET", "load cascade failed");
        return false;
    }
    // scaleFactor 在 detectMultiScale 调用时传入（1.3 / 1.1）
    impl_->ready = true;
    LOGI("CVDET", "CvDetector initialized (Haar cascade)");
    return true;
}

bool CvDetector::detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                        DetectionResult& result) {
    if (!impl_ || !impl_->ready) return false;

    // RGB888 -> BGR888 (OpenCV 默认 BGR)
    cv::Mat bgr(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
    for (uint32_t i = 0; i < w * h; ++i) {
        bgr.data[i * 3 + 0] = rgb[i * 3 + 2];  // B
        bgr.data[i * 3 + 1] = rgb[i * 3 + 1];  // G
        bgr.data[i * 3 + 2] = rgb[i * 3 + 0];  // R
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // 计算帧平均亮度（间隔采样）
    uint64_t bsum = 0;
    size_t bcnt = 0;
    for (int y = 0; y < static_cast<int>(h); y += 4) {
        const uint8_t* row = gray.ptr<uint8_t>(y);
        for (int x = 0; x < static_cast<int>(w); x += 4) {
            bsum += row[x];
            ++bcnt;
        }
    }
    uint8_t brightness = (bcnt > 0) ? static_cast<uint8_t>(bsum / bcnt) : 128;
    // 平滑
    last_brightness_ = static_cast<uint8_t>(last_brightness_ * 0.7f + brightness * 0.3f);
    bool is_night = (last_brightness_ < brightness_threshold_);

    // 夜间模式：先做 CLAHE 增强再检测
    if (is_night) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
        clahe->apply(gray, gray);
    } else {
        cv::equalizeHist(gray, gray);
    }

    // 1) 人脸检测
    // 夜间模式：放宽检测参数（减小 scaleFactor 和 minSize，增加召回率）
    std::vector<cv::Rect> faces;
    double scale_factor = is_night ? 1.15 : 1.3;
    int min_neighbors = is_night ? 2 : 3;
    cv::Size min_sz(is_night ? static_cast<int>(w) / 8 : static_cast<int>(w) / 6,
                    is_night ? static_cast<int>(h) / 8 : static_cast<int>(h) / 6);
    impl_->face_cascade.detectMultiScale(gray, faces, scale_factor, min_neighbors, 0, min_sz);

    if (faces.empty()) {
        // 无人脸：result.faces 为空，上层 PERCLOS 容错处理
        return true;
    }

    // 取最大的人脸（最近的人）
    auto it = std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });

    cv::Rect fr = *it;
    // 2) 在人脸 ROI 内做眼睛检测
    //    眼睛位于人脸上 1/3，缩小 ROI 提升稳定性、减少误检
    cv::Rect eye_region(fr.x, fr.y, fr.width, fr.height / 3);
    cv::Mat eye_roi = gray(eye_region);
    std::vector<cv::Rect> eyes;
    cv::Size eye_min(std::max(fr.width / 8, 16), std::max(fr.height / 12, 12));
    // minNeighbors=4 比默认 3 更严格，减少噪声框
    impl_->eye_cascade.detectMultiScale(eye_roi, eyes, 1.1, 4, 0, eye_min);

    // 单帧原始检测结果（Haar 睁眼时也会偶发漏检）
    bool raw_eyes_detected = !eyes.empty();

    // 时序去抖动：滑动窗口平滑，消除单帧漏检导致的睁闭眼抖动
    eye_history_.push_back(raw_eyes_detected);
    if (eye_history_.size() > kEyeHistoryLen) eye_history_.pop_front();
    size_t detect_cnt = 0;
    for (bool v : eye_history_) if (v) ++detect_cnt;

    // 三态判定（替代旧的否定式判定 eyes_closed = detect_cnt < thresh）
    //   detect_cnt >= kEyeOpenThresh → 睁眼，用眼睛框几何算连续 EAR
    //   detect_cnt == 0（连续全漏检）→ 推断闭眼
    //   1 <= detect_cnt < kEyeOpenThresh（偶发漏检）→ 不确定，保持上一帧 EAR，
    //   避免漏检直接判闭眼导致误报
    float ear;
    if (detect_cnt >= kEyeOpenThresh && !eyes.empty()) {
        // 睁眼：用本帧检测到的眼睛框几何算 EAR（6 点合成 → calc_ear）
        // Haar 眼睛框 cv::Rect(x,y,w,h)：EAR = h/w，睁眼≈0.30~0.50
        float ear_sum = 0;
        size_t n = 0;
        for (const auto& er : eyes) {
            Point2D p[6] = {
                {er.x + er.width * 0.00f, er.y + er.height * 0.5f},  // p0 左眼角
                {er.x + er.width * 0.33f, er.y + er.height * 0.0f},  // p1 上睑左
                {er.x + er.width * 0.66f, er.y + er.height * 0.0f},  // p2 上睑右
                {er.x + er.width * 1.00f, er.y + er.height * 0.5f},  // p3 右眼角
                {er.x + er.width * 0.66f, er.y + er.height * 1.0f},  // p4 下睑右
                {er.x + er.width * 0.33f, er.y + er.height * 1.0f},  // p5 下睑左
            };
            ear_sum += calc_ear(p);
            ++n;
        }
        ear = (n > 0) ? (ear_sum / static_cast<float>(n)) : last_ear_;
        last_ear_ = ear;
    } else if (detect_cnt == 0) {
        // 连续全漏检：推断闭眼
        ear = 0.10f;
        last_ear_ = ear;
    } else {
        // 偶发漏检（1 <= detect_cnt < kEyeOpenThresh）：保持上一帧，不主动判闭眼
        ear = last_ear_;
    }

    // 把检测计数编码进 score 字段，供 display 诊断显示（det=N/窗口）
    (void)detect_cnt;

    // 3) 构造 FaceBox
    FaceBox box;
    box.x = static_cast<float>(fr.x);
    box.y = static_cast<float>(fr.y);
    box.w = static_cast<float>(fr.width);
    box.h = static_cast<float>(fr.height);
    // score 编码"窗口内检测到眼睛的帧数"，供 display 诊断显示
    box.score = static_cast<float>(detect_cnt);
    box.ear_left  = ear;
    box.ear_right = ear;
    box.mar = 0.2f;  // Haar 无嘴部检测，给个中性值

    // 合成 5 个关键点（喂给 estimate_head_pose）
    // [0]左眼 [1]右眼 [2]鼻 [3]左嘴 [4]右嘴
    box.landmarks.resize(5);
    box.landmarks[0] = { box.x + box.w * 0.30f, box.y + box.h * 0.40f };  // 左眼
    box.landmarks[1] = { box.x + box.w * 0.70f, box.y + box.h * 0.40f };  // 右眼
    box.landmarks[2] = { box.x + box.w * 0.50f, box.y + box.h * 0.60f };  // 鼻
    box.landmarks[3] = { box.x + box.w * 0.35f, box.y + box.h * 0.75f };  // 左嘴
    box.landmarks[4] = { box.x + box.w * 0.65f, box.y + box.h * 0.75f };  // 右嘴

    estimate_head_pose(box.landmarks, w, h,
                       box.head_pitch, box.head_yaw, box.head_roll);

    result.faces.push_back(std::move(box));
    return true;
}

#else  // !DMS_HAS_OPENCV

// Stub：无 OpenCV 时返回空结果（保持流水线可运行）
struct CvDetector::Impl { bool ready = false; };

CvDetector::CvDetector(const std::string& model_dir)
    : model_dir_(model_dir), impl_(std::make_unique<Impl>()) {}
CvDetector::~CvDetector() = default;

std::string CvDetector::resolve_xml(const std::string&) const { return ""; }

bool CvDetector::init() {
    LOGE("CVDET", "[STUB] OpenCV not linked, CvDetector unavailable");
    return false;
}

bool CvDetector::detect(const uint8_t*, uint32_t, uint32_t,
                        DetectionResult& result) {
    (void)result;
    return false;
}

#endif  // DMS_HAS_OPENCV

}  // namespace dms
