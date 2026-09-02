#ifndef DMS_RKMPP_ENCODER_H
#define DMS_RKMPP_ENCODER_H

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

#include "utils/media_types.h"
#include "utils/ring_buffer.h"

// 前向声明，避免头文件强依赖 rockchip mpp 头
struct MppApi;
typedef struct MppCtx_t* MppCtx;
typedef struct MppBufferGroup_t* MppBufferGroup;
typedef struct MppBuffer_t* MppBuffer;
typedef struct MppFrame_t* MppFrame;
typedef struct MppPacket_t* MppPacket;

namespace dms {

// H.265 编码参数（事件触发录像场景调优）
struct EncoderConfig {
    uint32_t width       = 1280;
    uint32_t height      = 720;
    uint32_t fps         = 30;
    uint32_t gop         = 60;        // GOP = 2s@30fps，关键帧间隔
    uint32_t bitrate     = 2000000;   // 2 Mbps，车内场景够用
    uint32_t qp_init     = 26;
    uint32_t qp_min      = 10;
    uint32_t qp_max      = 51;
    PixelFormat input_fmt = PixelFormat::kNv12;  // VPU 输入 NV12
    uint32_t ring_capacity = 8;
};

// 视频编码回调：每编码出一帧 NALU 即回调上层（交给 Muxer）
using OnEncoded = std::function<void(const EncodedPacket& pkt)>;

// RKMPP H.265 硬件编码器
//
// 底层实现要点：
// 1. mpp_create → mpp_init(MPP_CTX_ENC, MPP_VIDEO_CodingHEVC)
// 2. 配置 mpi->control：RC_MODE / BITRATE / GOP / QP / 分辨率
// 3. 帧入队：mpp_frame 组装 NV12 → mpi->encode_put_frame
// 4. 码流出队：mpi->encode_get_packet → 拷贝 NALU → 回调上层
// 5. 编码线程内串行"入帧 → 出包"，保证 PTS 顺序
class RkmppEncoder {
public:
    RkmppEncoder();
    ~RkmppEncoder();

    RkmppEncoder(const RkmppEncoder&) = delete;
    RkmppEncoder& operator=(const RkmppEncoder&) = delete;

    // 初始化 MPP 编码上下文
    bool open(const EncoderConfig& cfg);

    // 注册码流回调
    void set_callback(OnEncoded cb) { on_encoded_ = std::move(cb); }

    // 喂一帧（NV12）入队；底层用环形缓冲解耦采集与编码
    // 返回 >0 表示因缓冲满而丢弃的帧数
    int push_frame(std::shared_ptr<VideoFrame> frame);

    // 启动编码线程
    bool start();

    // 停止并刷出剩余帧
    void stop();

    // 请求强制 IDR（事件触发录像时，确保新文件首帧为关键帧）
    void force_idr();

    uint64_t encoded_count() const { return encoded_.load(); }
    uint64_t input_dropped() const { return dropped_.load(); }

private:
    void encode_loop();
    bool encode_one(const VideoFrame& frame);

    EncoderConfig cfg_;
    std::unique_ptr<RingBuffer<std::shared_ptr<VideoFrame>>> ring_;
    std::thread       thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> want_idr_{false};
    std::atomic<uint64_t> encoded_{0};
    std::atomic<uint64_t> dropped_{0};
    OnEncoded         on_encoded_;

    // MPP 句柄（仅在 DMS_HAS_RKMPP 时使用）
    void* mpp_ctx_   = nullptr;  // MppCtx
    void* mpp_api_   = nullptr;  // MppApi*
    void* frame_grp_ = nullptr;  // MppBufferGroup（帧输入用）

    static constexpr const char* kTag = "ENC";
};

}  // namespace dms

#endif  // DMS_RKMPP_ENCODER_H
