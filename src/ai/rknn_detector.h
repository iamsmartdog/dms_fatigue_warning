#ifndef DMS_RKNN_DETECTOR_H
#define DMS_RKNN_DETECTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ai/detector.h"
#include "utils/media_types.h"

namespace dms {

// RKNN 人脸 + 关键点检测器
//
// 部署流程（在 PC 上完成，运行时只加载 .rknn）：
//   PyTorch/onnx 模型 → onnx → rknn-toolkit2 转换 + INT8 量化
//   输出：xxx.rknn（单文件，含 INT8 权重 + NPU 算子图）
//
// 模型约定（典型 RetinaFace/RFB 轻量版 + 关键点头）：
//   输入: 1 x 3 x H x W, RGB, 归一化 (x/255 - mean)/std
//   输出0: 1 x N x 5+   (cx, cy, w, h, score, ...)
//   输出1: 1 x N x 2*K  (每 anchor K 个关键点 x,y)
//
// 运行时工作流：
//   rknn_init → 设置 core mask（绑 NPU 核心）
//   循环：rknn_inputs_set → rknn_run → rknn_outputs_get
//   后处理：解码 anchor → 过滤 score → NMS → 计算 EAR/MAR/姿态
class RknnDetector : public Detector {
public:
    explicit RknnDetector(std::string model_path);
    ~RknnDetector() override;

    bool init() override;
    void input_shape(uint32_t& w, uint32_t& h) const override { w = in_w_; h = in_h_; }
    const char* name() const override { return "RknnDetector"; }

    bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                DetectionResult& result) override;

    // 配置
    struct Config {
        float score_thresh = 0.6f;
        float nms_thresh   = 0.4f;
        int   num_anchors  = 17640;     // 典型 RFB@320 输出 anchor 数
        int   num_landmarks = 5;        // 5 点（左眼/右眼/鼻/左嘴/右嘴）
        // 预处理：mean/std（INT8 量化前）
        float mean[3] = {127.0f, 127.0f, 127.0f};
        float std[3]  = {128.0f, 128.0f, 128.0f};
        // NPU 核心：0=auto, 1=core0, 2=core1, 3=core2
        int core_mask = 0;
    };
    void set_config(const Config& c) { cfg_ = c; }

private:
    // 推理：把 rgb 拷进 input tensor，调用 rknn_run
    bool run_inference(const uint8_t* rgb);

    // 后处理：解码 anchor → 过滤 → NMS → 填充 FaceBox
    void postprocess(DetectionResult& out);

    std::string model_path_;
    Config cfg_;
    uint32_t in_w_ = 320;
    uint32_t in_h_ = 240;

    void* rknn_ctx_ = nullptr;  // rknn_context
    bool  initialized_ = false;

    // 推理输出缓存（resize 于 init）
    std::vector<float> out_bboxes_;   // [num_anchors][5+]
    std::vector<float> out_kps_;      // [num_anchors][2*num_landmarks]
};

}  // namespace dms

#endif  // DMS_RKNN_DETECTOR_H
