#include "ai/pfld_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "utils/log.h"

namespace dms {

// ===== PFLDDetector::Impl =====
struct PFLDDetector::Impl {
    cv::dnn::Net net;
    bool loaded = false;

    bool init(const std::string& path) {
        try {
            net = cv::dnn::readNetFromONNX(path);
            if (net.empty()) {
                LOGE("PFLD", "Failed to read ONNX model: %s", path.c_str());
                return false;
            }
            // 设置为 CPU 推理
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            loaded = true;
            LOGI("PFLD", "Model loaded: %s", path.c_str());
            return true;
        } catch (const std::exception& e) {
            LOGE("PFLD", "Exception loading model: %s", e.what());
            return false;
        }
    }

    bool detect(const cv::Mat& face_bgr, std::vector<float>& landmarks) {
        if (!loaded) {
            LOGE("PFLD", "Model not loaded");
            return false;
        }

        try {
            // 缩放人脸到 112x112
            cv::Mat input;
            cv::resize(face_bgr, input, cv::Size(112, 112));

            // 转 blob：归一化到 [0,1]，PFLD 需要 BGR 输入
            cv::Mat blob = cv::dnn::blobFromImage(
                input, 1.0 / 255.0, cv::Size(112, 112),
                cv::Scalar(0, 0, 0), true, false);

            net.setInput(blob);
            cv::Mat output = net.forward();

            // PFLD 输出：1x212 或 1x106x2 取决于模型
            // 常见格式：1x212（先全部 x，再全部 y）或 1x212（x,y,x,y...交替）
            // 或者 1x106x2（每个关键点两个坐标）
            landmarks.clear();
            if (output.total() == 212) {
                // 展平格式：1x212
                // PFLD 标准输出：前 106 个是 x，后 106 个是 y
                const float* data = output.ptr<float>();
                landmarks.resize(212);
                for (int i = 0; i < 106; ++i) {
                    landmarks[i * 2] = data[i];           // x
                    landmarks[i * 2 + 1] = data[i + 106]; // y
                }
            } else if (output.size[1] == 106 && output.size[2] == 2) {
                // 1x106x2 格式
                const float* data = output.ptr<float>();
                landmarks.resize(212);
                for (int i = 0; i < 106; ++i) {
                    landmarks[i * 2] = data[i * 2];           // x
                    landmarks[i * 2 + 1] = data[i * 2 + 1];   // y
                }
            } else {
                LOGE("PFLD", "Unexpected output shape: %zu total elements",
                     output.total());
                return false;
            }

            return true;
        } catch (const std::exception& e) {
            LOGE("PFLD", "Exception during detect: %s", e.what());
            return false;
        }
    }
};

PFLDDetector::PFLDDetector(const std::string& model_path)
    : model_path_(model_path), impl_(std::make_unique<Impl>()) {}

PFLDDetector::~PFLDDetector() = default;

bool PFLDDetector::init() {
    if (model_path_.empty()) {
        LOGE("PFLD", "Model path not set");
        return false;
    }
    return impl_->init(model_path_);
}

bool PFLDDetector::detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                           const FaceBox& bbox, std::vector<Point2D>& landmarks) {
    if (!rgb || w == 0 || h == 0) {
        LOGE("PFLD", "Invalid input");
        return false;
    }

    try {
        // 裁剪人脸区域
        int x = static_cast<int>(bbox.x);
        int y = static_cast<int>(bbox.y);
        int fw = static_cast<int>(bbox.w);
        int fh = static_cast<int>(bbox.h);

        // 确保裁剪区域不越界
        x = std::max(0, x);
        y = std::max(0, y);
        fw = std::min(fw, static_cast<int>(w) - x);
        fh = std::min(fh, static_cast<int>(h) - y);

        if (fw <= 0 || fh <= 0) {
            LOGE("PFLD", "Invalid face region: %d,%d %dx%d", x, y, fw, fh);
            return false;
        }

        // 创建 OpenCV Mat 并裁剪
        cv::Mat rgb_img(h, w, CV_8UC3, (void*)rgb);
        cv::Mat face_roi = rgb_img(cv::Rect(x, y, fw, fh)).clone();

        // 转 BGR 给 PFLD 模型
        cv::Mat face_bgr;
        cv::cvtColor(face_roi, face_bgr, cv::COLOR_RGB2BGR);

        // 推理
        std::vector<float> raw_lm;
        if (!impl_->detect(face_bgr, raw_lm)) {
            return false;
        }

        // 解析结果
        landmarks.clear();
        landmarks.resize(106);

        // PFLD 输出坐标归一化到 [0,1]（相对于 112x112 人脸区域）
        // 需要映射回原图坐标
        for (int i = 0; i < 106; ++i) {
            float nx = raw_lm[i];
            float ny = raw_lm[i + 106];  // 后 106 个是 y 坐标

            // 映射回原图
            landmarks[i].x = nx * fw + x;
            landmarks[i].y = ny * fh + y;
        }

        // 调试：打印前几个 landmark 验证坐标范围
        static int dbg_cnt = 0;
        if (++dbg_cnt % 30 == 0) {
            LOGI("PFLD", "[DBG] raw_lm[0]=%.4f raw_lm[1]=%.4f raw_lm[106]=%.4f raw_lm[107]=%.4f "
                 "-> lm[0]=(%.1f,%.1f) lm[1]=(%.1f,%.1f) face_region=(%d,%d,%dx%d)",
                 raw_lm[0], raw_lm[1], raw_lm[106], raw_lm[107],
                 landmarks[0].x, landmarks[0].y,
                 landmarks[1].x, landmarks[1].y,
                 x, y, fw, fh);
        }

        return true;
    } catch (const std::exception& e) {
        LOGE("PFLD", "Exception: %s", e.what());
        return false;
    }
}

}  // namespace dms