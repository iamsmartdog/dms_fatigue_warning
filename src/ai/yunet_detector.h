#ifndef DMS_YUNET_DETECTOR_H
#define DMS_YUNET_DETECTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ai/detector.h"
#include "ai/pfld_detector.h"

namespace dms {

// YuNet 深度学习人脸检测器（OpenCV DNN）
//
// 用途：
// 1. 基于 OpenCV 的 FaceDetectorYN（YuNet 模型）做人脸检测
// 2. 对低光环境有更好的鲁棒性（相比 Haar 级联）
// 3. 输出人脸 bbox + 5 点关键点
//
// 模型说明：
// - 输入：RGB888 或 BGR888 图像，任意尺寸（内部自动缩放到 320x320）
// - 输出：人脸 bbox + 5 点关键点（眼睛、鼻、嘴）+ 置信度
// - 所需库：OpenCV >= 4.8.0 with FaceDetectorYN
//
// 检测精度（WIDER Face 数据集）：
// - Easy: 0.884, Medium: 0.866, Hard: 0.750

class YuNetDetector : public Detector {
public:
    // model_path: YuNet ONNX 模型文件路径（.onnx）
    // pfld_path: PFLD 106 点 ONNX 模型路径（可选）
    explicit YuNetDetector(const std::string& model_path = "",
                           const std::string& pfld_path = "");
    ~YuNetDetector() override;

    bool init() override;

    void input_shape(uint32_t& w, uint32_t& h) const override {
        w = model_w_;
        h = model_h_;
    }

    // rgb: 输入图像 RGB888，w*h*3
    // result: 输出检测结果（含 EAR/姿态等）
    bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                DetectionResult& result) override;

    const char* name() const override { return "YuNetDetector"; }

private:
    std::string model_path_;
    std::string pfld_model_path_;
    uint32_t model_w_ = 320;
    uint32_t model_h_ = 320;

    // PFLD 关键点检测器
    std::unique_ptr<PFLDDetector> pfld_;

    // PIMPL：隐藏 OpenCV 的实现细节
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dms

#endif  // DMS_YUNET_DETECTOR_H
