#ifndef DMS_PFLD_DETECTOR_H
#define DMS_PFLD_DETECTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "utils/media_types.h"

namespace dms {

// PFLD 106 点人脸关键点检测器
// 用途：
// 1. 基于 ONNX 模型的 106 个人脸关键点检测
// 2. 输入：人脸区域 RGB 图像 + bbox
// 3. 输出：106 个归一化坐标点 (0~1)，需要映射回原图
//
// 关键点索引对应关系（PFLD 标准）：
// - [0-19]:   左眉毛（20 个点）
// - [20-39]:  右眉毛（20 个点）
// - [40-59]:  左眼睛（20 个点）
// - [60-79]:  右眼睛（20 个点）
// - [80-87]:  鼻子（8 个点）
// - [88-95]:  左嘴角（8 个点）
// - [96-103]: 右嘴角（8 个点）
// - [104-105]: 下巴中心（2 个点）

class PFLDDetector {
public:
    explicit PFLDDetector(const std::string& model_path = "");
    ~PFLDDetector();

    bool init();

    // 输入：
    // - face_img: 人脸区域图像（RGB888），通常是 112x112
    // - bbox: 人脸在原图的位置（用于坐标映射）
    // 输出：
    // - landmarks: 106 个点，坐标在原图空间
    bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                const FaceBox& bbox, std::vector<Point2D>& landmarks);

private:
    std::string model_path_;
    uint32_t model_w_ = 112;
    uint32_t model_h_ = 112;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dms

#endif  // DMS_PFLD_DETECTOR_H
