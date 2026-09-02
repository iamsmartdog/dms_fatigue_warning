#ifndef DMS_CV_DETECTOR_H
#define DMS_CV_DETECTOR_H

#include <cstdint>
#include <deque>
#include <memory>
#include <string>

#include "ai/detector.h"

namespace dms {

// OpenCV Haar 级联检测器（开发机/虚拟机可用，无需 NPU）
//
// 用途：
// 1. 在虚拟机/开发机上跑通真实人脸 + 闭眼检测，验证整条流水线
// 2. 上板后由 RknnDetector 接管；本类不依赖任何板端库
//
// 实现要点：
// - 人脸检测：cv::CascadeClassifier (haarcascade_frontalface_alt2)
// - 闭眼判定（三态，替代旧的否定式判定，避免漏检误判闭眼）：
//     detect_cnt >= kEyeOpenThresh → 睁眼，用眼睛框几何算连续 EAR
//     detect_cnt == 0（连续全漏检）→ 推断闭眼
//     1 <= detect_cnt < kEyeOpenThresh（偶发漏检）→ 不确定，保持上一帧 EAR
// - EAR：用眼睛框 6 点合成调用 calc_ear（=h/w，睁眼≈0.30~0.50），
//   漏检/不确定时保持 last_ear_，不再写死二值 0.10/0.30
// - 关键点：Haar 无关键点输出，按人脸框几何合成 5 点喂给 estimate_head_pose
//
// 模型文件搜索顺序（兼容 apt 装的 opencv 与源码编译的 opencv）：
//   1. 命令行 -m 指定的目录
//   2. /usr/share/opencv4/haarcascades/
//   3. /usr/share/opencv/haarcascades/
//   4. /usr/local/share/opencv4/haarcascades/
//   5. /usr/local/src/opencv-4.12.0/data/haarcascades/  （源码编译默认路径）
class CvDetector : public Detector {
public:
    // model_dir: haarcascade_*.xml 所在目录；空则自动搜索
    explicit CvDetector(const std::string& model_dir = "");
    ~CvDetector() override;

    bool init() override;

    void input_shape(uint32_t& w, uint32_t& h) const override {
        w = model_w_; h = model_h_;
    }

    // rgb: 输入图像 RGB888，w*h*3
    // result: 输出检测结果（含 EAR/姿态等）
    bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                DetectionResult& result) override;

    const char* name() const override { return "CvDetector"; }

private:
    // 在候选目录里查找 haarcascade xml，返回完整路径
    std::string resolve_xml(const std::string& basename) const;

    std::string model_dir_;          // 用户指定的目录
    // Haar 眼睛检测对分辨率敏感：320x240 下眼睛区域仅~30x15px，睁眼也频繁漏检
    // 提升到 640x480，眼睛区域~60x30px，检测稳定得多
    uint32_t   model_w_ = 640;
    uint32_t   model_h_ = 480;

    // 时序去抖动：记录最近 N 帧的眼睛检测状态，消除单帧漏检导致的睁闭眼抖动
    // Haar 眼睛检测器在睁眼时也会偶发漏检，需要滑动窗口平滑
    // 窗口15帧(10fps≈1.5s)，检测到≥5帧就判睁眼 → 容忍 66% 漏检率
    // 只有持续闭眼(几乎100%漏检)才会判定闭眼
    std::deque<bool> eye_history_;   // true=本帧检测到眼睛
    static constexpr size_t kEyeHistoryLen   = 15;  // 窗口长度（10fps≈1.5s）
    static constexpr size_t kEyeOpenThresh   = 5;   // 窗口内"检测到眼睛"帧数≥此值就判睁眼
    float last_ear_ = 0.35f;          // 上一帧 EAR，漏检/不确定时保持（默认睁眼值）
    
    // 夜间模式参数调整
    uint8_t last_brightness_ = 128;   // 上一帧亮度
    uint8_t brightness_threshold_ = 80;  // 亮度阈值（低于此值时激活夜间模式）

    struct Impl;                     // PIMPL，避免头文件依赖 opencv
    std::unique_ptr<Impl> impl_;
};

}  // namespace dms

#endif  // DMS_CV_DETECTOR_H
