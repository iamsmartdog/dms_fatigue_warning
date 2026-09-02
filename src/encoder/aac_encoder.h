#ifndef DMS_AAC_ENCODER_H
#define DMS_AAC_ENCODER_H

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <thread>

#include "utils/media_types.h"
#include "utils/ring_buffer.h"

// 前向声明 faac / fdkaac 句柄
namespace dms {

// AAC 音频编码器
//
// 选型说明：
// - 嵌入式 Linux 上常用 libfdk-aac（音质最好）或 faac（更轻量）
// - 这里以 fdkaac 的 API 为蓝本，宏 DMS_HAS_FDKAAC 包裹；
//   若链接 faac 则改 DMS_HAS_FAAC 分支
// - 编码线程从输入环形缓冲消费 PCM，输出 AAC ADTS 帧
class AacEncoder {
public:
    // 编码完成回调（输出 AAC 帧）
    using OnEncoded = std::function<void(const EncodedPacket&)>;

    AacEncoder();
    ~AacEncoder();

    AacEncoder(const AacEncoder&) = delete;
    AacEncoder& operator=(const AacEncoder&) = delete;

    // 初始化编码器
    bool open(const AudioConfig& cfg);

    void set_callback(OnEncoded cb) { on_encoded_ = std::move(cb); }

    // 喂一帧 PCM（内部按 1024 样本/AAC 帧分块）
    int push_frame(const AudioFrame& frame);

    bool start();
    void stop();

    // AAC AudioSpecificConfig（MP4 esds box 需要，2 字节 LC profile）
    void get_audio_specific_config(uint8_t asc[2]) const;

    uint64_t encoded_count() const { return encoded_.load(); }

private:
    void encode_loop();
    bool encode_block(const int16_t* pcm, size_t frames,
                      uint64_t pts_ms, bool is_first);

    AudioConfig cfg_;
    std::unique_ptr<RingBuffer<AudioFrame>> ring_;
    std::thread      thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> encoded_{0};
    OnEncoded on_encoded_;

    // 编码器状态（仅在 DMS_HAS_FDKAAC / FAAC 时有效）
    void* handle_ = nullptr;
    unsigned long input_samples_ = 0;   // 每次 encode 需要的样本数（典型 1024）
    unsigned long max_output_bytes_ = 0;
    int aac_object_ = 2;   // LC
    uint8_t asc_[2] = {0};

    // 输入缓冲：累积到一帧 AAC 所需样本数再编码
    std::vector<int16_t> pcm_buf_;
    uint64_t buf_pts_base_ms_ = 0;
    uint64_t buf_samples_filled_ = 0;
    bool first_block_ = true;

    static constexpr const char* kTag = "AAC";
};

}  // namespace dms

#endif  // DMS_AAC_ENCODER_H
