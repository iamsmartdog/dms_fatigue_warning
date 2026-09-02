#ifndef DMS_MEDIA_TYPES_H
#define DMS_MEDIA_TYPES_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ring_buffer.h"

namespace dms {

// ============================================================
// 视频相关
// ============================================================

// 像素格式（与 V4L2 FourCC 对应）
enum class PixelFormat : uint32_t {
    kYuyv = 0x56595559,  // 'YUYV' YUV422 packed
    kMjpg  = 0x47504A4D,  // 'MJPG' Motion JPEG
    kNv12  = 0x3231564E,  // 'NV12' YUV420 semi-planar（RK ISP / RKMPP 输入）
    kRgb888 = 0           // 仅内部使用，非 FourCC
};

// 由外部释放策略决定的帧内存所有权类型
enum class FrameMemory : uint8_t {
    kExternal = 0,  // 不持有，data 指向 mmap / DMA-BUF 等外部内存
    kOwned    = 1,  // 持有，data 为 new[] 分配
};

// 视频帧（扩展自 utils/ring_buffer.h 中的 Frame）
// 这里给出一个"自包含"版本：可选持有数据所有权，便于跨线程深度拷贝
struct VideoFrame {
    uint8_t*      data       = nullptr;
    size_t        size       = 0;       // data 字节数
    uint32_t      width      = 0;
    uint32_t      height     = 0;
    PixelFormat   format     = PixelFormat::kMjpg;
    uint64_t      pts_ms     = 0;       // 毫秒级 PTS（音视频同步基准）
    uint32_t      seq        = 0;       // 序列号
    bool          is_valid   = false;
    FrameMemory   memory     = FrameMemory::kExternal;

    VideoFrame() = default;

    // 深拷贝构造（用于把 mmap 帧拷出采集线程）
    void copy_from(const uint8_t* src, size_t n) {
        if (memory == FrameMemory::kOwned) delete[] data;
        data = new uint8_t[n];
        std::memcpy(data, src, n);
        size = n;
        memory = FrameMemory::kOwned;
    }

    ~VideoFrame() {
        if (memory == FrameMemory::kOwned) delete[] data;
    }

    VideoFrame(const VideoFrame&) = delete;
    VideoFrame& operator=(const VideoFrame&) = delete;

    // 移动语义
    VideoFrame(VideoFrame&& o) noexcept { *this = std::move(o); }
    VideoFrame& operator=(VideoFrame&& o) noexcept {
        if (this != &o) {
            if (memory == FrameMemory::kOwned) delete[] data;
            data = o.data; size = o.size; width = o.width; height = o.height;
            format = o.format; pts_ms = o.pts_ms; seq = o.seq;
            is_valid = o.is_valid; memory = o.memory;
            o.data = nullptr; o.size = 0; o.memory = FrameMemory::kExternal;
        }
        return *this;
    }
};

// ============================================================
// 音频相关
// ============================================================

// 音频参数
struct AudioConfig {
    uint32_t sample_rate = 16000;  // 重采样后的目标采样率（喂给编码器）
    uint32_t channels    = 1;      // 单声道
    uint32_t format_bits = 16;     // S16_LE
    uint32_t period_frames = 1024; // ALSA 周期帧数
    std::string device = "default";
};

// 音频帧（PCM 样本）
struct AudioFrame {
    std::vector<uint8_t> data;     // PCM 数据
    uint64_t pts_ms = 0;
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
    uint32_t format_bits = 0;
};

// ============================================================
// 编码后码流包
// ============================================================

enum class MediaType : uint8_t { kVideo, kAudio };

struct EncodedPacket {
    std::vector<uint8_t> data;     // H.265 NALU / AAC 帧
    MediaType type = MediaType::kVideo;
    uint64_t pts_ms = 0;
    uint64_t dts_ms = 0;
    bool    keyframe = false;      // 视频关键帧
};

// ============================================================
// AI 检测结果
// ============================================================

// 2D 点（人脸关键点）
struct Point2D { float x = 0; float y = 0; };

// 单张人脸检测结果
struct FaceBox {
    float x = 0, y = 0, w = 0, h = 0;   // 像素坐标 bbox
    float score = 0;                    // 检测置信度
    std::vector<Point2D> landmarks;     // 关键点（眼睛/鼻/嘴）
    float ear_left = 0;                 // 左眼纵横比 Eye Aspect Ratio
    float ear_right = 0;                // 右眼 EAR
    float mar = 0;                      // 嘴部纵横比 Mouth Aspect Ratio
    float head_pitch = 0;               // 头部俯仰角（度）
    float head_yaw = 0;
    float head_roll = 0;
};

struct DetectionResult {
    std::vector<FaceBox> faces;
    uint64_t pts_ms = 0;        // 对应原始帧的 PTS
    int inference_ms = 0;       // 本次推理耗时
};

// ============================================================
// 预警事件
// ============================================================

enum class AlarmLevel : uint8_t {
    kNone   = 0,
    kInfo   = 1,   // 提示（连续眨眼）
    kWarn   = 2,   // 警告（轻度疲劳）
    kDanger = 3,   // 危险（重度疲劳/闭眼过长）
};

inline const char* alarm_level_str(AlarmLevel lv) {
    switch (lv) {
        case AlarmLevel::kInfo:   return "INFO";
        case AlarmLevel::kWarn:   return "WARN";
        case AlarmLevel::kDanger: return "DANGER";
        default: return "NONE";
    }
}

struct AlarmEvent {
    AlarmLevel level = AlarmLevel::kNone;
    std::string reason;          // 触发原因
    uint64_t pts_ms = 0;         // 触发时刻
    float metric = 0;            // 关键指标（PERCLOS/EAR 等）

    // 用于触发事件录像：事件前后窗口
    uint64_t pre_roll_ms  = 5000;  // 前 5s
    uint64_t post_roll_ms = 5000;  // 后 5s
};

}  // namespace dms

#endif  // DMS_MEDIA_TYPES_H
