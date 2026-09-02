#ifndef DMS_IMAGE_PREPROCESS_H
#define DMS_IMAGE_PREPROCESS_H

#include <cstdint>
#include <memory>
#include <vector>

#include "utils/media_types.h"

namespace dms {

// 图像预处理（USB 摄像头场景下，用软件替代 RK ISP）
//
// 职责（对应设计文档中的"ISP/预处理"线程）：
// 1. 解码：MJPG → RGB888（libjpeg-turbo）/ YUYV → RGB888（直接矩阵变换）
// 2. resize：双线性缩放到模型输入尺寸（如 320x240）
// 3. 像素格式转换：RGB888 → NV12（VPU 编码要求）
// 4. 基础增强（USB 摄像头无法走 RK ISP，用软件近似）：
//    - 直方图均衡/CLAHE 风格的轻量对比度拉伸，缓解逆光
//    - 简单时域降噪（与上一帧加权融合），近似 3DNR
//
// 设计上每个函数都是纯函数（无内部状态，线程安全），可被多线程调用；
// 时域降噪类 Preprocessor 内部保留上一帧（单实例非线程安全，由
// 流水线保证同一 Preprocessor 仅被一个 preprocess 线程驱动）
class ImagePreprocess {
public:
    ImagePreprocess() = default;
    ~ImagePreprocess() = default;

    // 配置：输入格式 + 目标编码分辨率 + 目标模型输入分辨率
    struct Config {
        uint32_t enc_width  = 1280;
        uint32_t enc_height = 720;
        uint32_t model_width  = 320;
        uint32_t model_height = 240;
        bool enable_denoise = true;     // 软件时域降噪（近似 3DNR）
        float denoise_alpha = 0.75f;    // 上一帧权重 (0~1)，越大越平滑
        bool enable_contrast_enhance = true;
        bool enable_clahe = true;       // 自适应直方图均衡，低光增强
        bool enable_adaptive_denoise = true;  // 根据亮度自适应降噪强度
        uint8_t brightness_threshold = 80;    // 亮度阈值，低于此值激活夜间模式
    };

    void set_config(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    // ============== 解码到 RGB888 ==============
    // MJPG → RGB888（调用 libjpeg）
    bool decode_mjpg_to_rgb(const uint8_t* jpg, size_t size,
                            std::vector<uint8_t>& rgb,
                            uint32_t& w, uint32_t& h);
    // YUYV → RGB888（单线程纯计算，无需第三方库）
    void yuyv_to_rgb(const uint8_t* yuyv, uint32_t w, uint32_t h,
                     std::vector<uint8_t>& rgb);

    // ============== 颜色空间转换 ==============
    // RGB888 → NV12（喂给 VPU / RKNN）
    void rgb_to_nv12(const uint8_t* rgb, uint32_t w, uint32_t h,
                     std::vector<uint8_t>& nv12);

    // ============== resize（双线性） ==============
    void resize_bilinear(const uint8_t* src, uint32_t sw, uint32_t sh,
                         uint8_t* dst, uint32_t dw, uint32_t dh,
                         uint32_t channels = 3);

    // ============== 增强（软件近似 ISP） ==============
    // 对比度拉伸（直方图 1% 段拉伸），缓解逆光
    void contrast_enhance(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h);
    // 自适应直方图均衡（CLAHE 风格），低光增强
    void clahe_enhance(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h);
    // 计算帧平均亮度
    uint8_t compute_brightness(const std::vector<uint8_t>& rgb, uint32_t w, uint32_t h);
    // 时域降噪：与上一帧加权融合（首次调用直通）
    void temporal_denoise(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h);

    // ============== 一站式：原始帧 → 编码帧(NV12) + 模型输入(RGB) ==============
    struct PreprocResult {
        std::shared_ptr<VideoFrame> enc_frame;   // 给编码器（NV12）
        std::vector<uint8_t> model_input_rgb;    // 给 AI 检测（resize 后 RGB）
        uint32_t model_w = 0;
        uint32_t model_h = 0;
        bool ok = false;
    };

    // 输入：从 V4L2 取到的原始帧（MJPG/YUYV）
    PreprocResult process(const VideoFrame& raw);

private:
    Config cfg_;
    std::vector<uint8_t> prev_frame_;  // 上一帧 RGB（时域降噪用）
    bool has_prev_ = false;
    uint8_t last_brightness_ = 128;    // 上一帧亮度（用于平滑）
    bool is_night_mode_ = false;       // 是否处于夜间模式
};

}  // namespace dms

#endif  // DMS_IMAGE_PREPROCESS_H
