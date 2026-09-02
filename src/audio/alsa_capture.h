#ifndef DMS_ALSA_CAPTURE_H
#define DMS_ALSA_CAPTURE_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "utils/media_types.h"
#include "utils/ring_buffer.h"

// 前向声明，避免头文件强依赖 <alsa/asoundlib.h>
struct _snd_pcm;
typedef struct _snd_pcm snd_pcm_t;

namespace dms {

// ALSA 音频采集器
//
// 底层实现要点：
// 1. snd_pcm_open/hw_params 配置 PCM 采集（采样率/格式/通道/周期）
// 2. 内部采集线程 snd_pcm_readi 周期性读取 PCM 帧
// 3. 若硬件采样率 != 目标采样率，做线性重采样适配编码器
// 4. 帧带 PTS 时间戳，与 V4L2 处于同一时钟域（系统单调时钟 ms）
class AlsaCapture {
public:
    AlsaCapture();
    ~AlsaCapture();

    AlsaCapture(const AlsaCapture&) = delete;
    AlsaCapture& operator=(const AlsaCapture&) = delete;

    // 打开 PCM 设备并配置 hw_params
    bool open(const AudioConfig& cfg);

    // 启动采集线程
    bool start();

    // 停止采集
    void stop();

    // 取一帧 PCM（阻塞）
    bool get_frame(AudioFrame& frame, int timeout_ms = -1);

    uint64_t captured_count() const { return captured_.load(); }
    uint64_t dropped_count()   const { return dropped_.load(); }
    uint32_t hardware_rate()   const { return hw_rate_; }
    uint32_t hardware_channels() const { return hw_channels_; }

private:
    void capture_loop();
    bool setup_hw_params();

    // 线性插值重采样（单声道/双声道 S16），将 hw 帧重采样到目标采样率
    void resample_s16(const int16_t* src, uint32_t src_rate, size_t src_frames,
                      std::vector<int16_t>& dst, uint32_t dst_rate);

    snd_pcm_t*  pcm_        = nullptr;
    AudioConfig cfg_;
    uint32_t    hw_rate_    = 0;
    uint32_t    hw_channels_ = 0;
    uint32_t    hw_format_bytes_ = 2;   // S16 = 2 bytes/sample

    std::unique_ptr<RingBuffer<AudioFrame>> ring_;
    std::thread        thread_;
    std::atomic<bool>  running_{false};
    std::atomic<uint64_t> captured_{0};
    std::atomic<uint64_t> dropped_{0};

    // PTS 累计：已输出到 ring 的样本数（按目标采样率计）
    uint64_t output_samples_ = 0;
    uint64_t start_ms_ = 0;

    static constexpr const char* kTag = "ALSA";
};

}  // namespace dms

#endif  // DMS_ALSA_CAPTURE_H
