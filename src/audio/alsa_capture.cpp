#include "audio/alsa_capture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cmath>

#ifdef DMS_HAS_ALSA
#include <alsa/asoundlib.h>
#endif

#include "utils/log.h"

namespace dms {

#ifdef DMS_HAS_ALSA

// ======================== 真实 ALSA 实现 ========================

AlsaCapture::AlsaCapture() = default;

AlsaCapture::~AlsaCapture() {
    stop();
    if (pcm_) {
        snd_pcm_drain(pcm_);
        snd_pcm_close(pcm_);
        pcm_ = nullptr;
    }
}

bool AlsaCapture::setup_hw_params() {
    snd_pcm_hw_params_t* hp = nullptr;
    snd_pcm_hw_params_alloca(&hp);

    if (snd_pcm_hw_params_any(pcm_, hp) < 0) {
        LOGE(kTag, "snd_pcm_hw_params_any failed");
        return false;
    }

    // 交错读写访问
    if (snd_pcm_hw_params_set_access(pcm_, hp, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        LOGE(kTag, "set_access failed");
        return false;
    }

    // 格式：S16_LE
    snd_pcm_format_t fmt = SND_PCM_FORMAT_S16_LE;
    if (snd_pcm_hw_params_set_format(pcm_, hp, fmt) < 0) {
        LOGE(kTag, "set_format S16_LE failed");
        return false;
    }
    hw_format_bytes_ = 2;

    // 优先请求目标采样率，硬件不支持时取邻近值
    uint32_t rate = cfg_.sample_rate;
    int dir = 0;
    if (snd_pcm_hw_params_set_rate_near(pcm_, hp, &rate, &dir) < 0) {
        LOGE(kTag, "set_rate_near failed");
        return false;
    }
    hw_rate_ = rate;

    // 通道
    uint32_t ch = cfg_.channels;
    if (snd_pcm_hw_params_set_channels_near(pcm_, hp, &ch) < 0) {
        LOGE(kTag, "set_channels failed");
        return false;
    }
    hw_channels_ = ch;

    // 周期大小
    snd_pcm_uframes_t period = cfg_.period_frames;
    dir = 0;
    if (snd_pcm_hw_params_set_period_size_near(pcm_, hp, &period, &dir) < 0) {
        LOGE(kTag, "set_period_size_near failed");
        return false;
    }
    // buffer = 4 个周期
    snd_pcm_uframes_t buffer = period * 4;
    if (snd_pcm_hw_params_set_buffer_size_near(pcm_, hp, &buffer) < 0) {
        LOGE(kTag, "set_buffer_size_near failed");
        return false;
    }

    if (snd_pcm_hw_params(pcm_, hp) < 0) {
        LOGE(kTag, "snd_pcm_hw_params failed");
        return false;
    }

    LOGI(kTag, "hw: rate=%u ch=%u period=%lu buffer=%lu",
         hw_rate_, hw_channels_, (unsigned long)period, (unsigned long)buffer);
    return true;
}

bool AlsaCapture::open(const AudioConfig& cfg) {
    cfg_ = cfg;

    int err = snd_pcm_open(&pcm_, cfg_.device.c_str(),
                           SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        LOGE(kTag, "snd_pcm_open(%s) failed: %s",
             cfg_.device.c_str(), snd_strerror(err));
        return false;
    }

    if (!setup_hw_params()) {
        snd_pcm_close(pcm_);
        pcm_ = nullptr;
        return false;
    }

    ring_ = std::make_unique<RingBuffer<AudioFrame>>(8);
    LOGI(kTag, "alsa capture opened, target=%uch%u@%uHz",
         cfg_.channels, cfg_.channels, cfg_.sample_rate);
    return true;
}

bool AlsaCapture::start() {
    if (running_.exchange(true)) return true;
    if (!pcm_) {
        LOGE(kTag, "pcm not opened");
        running_ = false;
        return false;
    }

    int err = snd_pcm_prepare(pcm_);
    if (err < 0) {
        LOGE(kTag, "snd_pcm_prepare: %s", snd_strerror(err));
        running_ = false;
        return false;
    }
    start_ms_ = now_ms();
    output_samples_ = 0;
    captured_ = 0;
    dropped_ = 0;
    ring_->reset();
    thread_ = std::thread(&AlsaCapture::capture_loop, this);
    LOGI(kTag, "capture thread started");
    return true;
}

void AlsaCapture::stop() {
    if (!running_.exchange(false)) return;
    // 先 drop PCM 使阻塞的 snd_pcm_readi 返回，再 join（否则 join 永久挂死）
    if (pcm_) snd_pcm_drop(pcm_);
    if (ring_) ring_->stop();
    if (thread_.joinable()) thread_.join();
    LOGI(kTag, "stopped, captured=%lu dropped=%lu",
         captured_.load(), dropped_.load());
}

void AlsaCapture::resample_s16(const int16_t* src, uint32_t src_rate, size_t src_frames,
                               std::vector<int16_t>& dst, uint32_t dst_rate) {
    if (src_rate == dst_rate) {
        // 直通拷贝
        dst.assign(src, src + src_frames * hw_channels_);
        return;
    }
    // 线性插值（src_frames-1 个区间，每区间插入 (dst_rate/src_rate - 1) 点）
    const double ratio = static_cast<double>(dst_rate) / src_rate;
    const size_t out_frames = static_cast<size_t>(src_frames * ratio);
    dst.resize(out_frames * hw_channels_);
    for (size_t i = 0; i < out_frames; ++i) {
        double src_pos = i / ratio;
        size_t idx = static_cast<size_t>(src_pos);
        double frac = src_pos - idx;
        if (idx + 1 >= src_frames) idx = src_frames - 2;
        for (uint32_t c = 0; c < hw_channels_; ++c) {
            double v = src[idx * hw_channels_ + c] * (1.0 - frac) +
                       src[(idx + 1) * hw_channels_ + c] * frac;
            dst[i * hw_channels_ + c] = static_cast<int16_t>(std::lround(v));
        }
    }
}

void AlsaCapture::capture_loop() {
    LOGI(kTag, "capture_loop tid=%u", static_cast<unsigned>(::gettid()));

    const snd_pcm_uframes_t period = cfg_.period_frames;
    std::vector<int16_t>   raw(period * hw_channels_);
    std::vector<int16_t>   resampled;

    while (running_.load()) {
        snd_pcm_sframes_t n = snd_pcm_readi(pcm_, raw.data(), period);
        if (n < 0) {
            // underrun / 恢复
            LOGW(kTag, "snd_pcm_readi: %s, recovering", snd_strerror(n));
            n = snd_pcm_recover(pcm_, static_cast<int>(n), 1);
            if (n < 0) {
                LOGE(kTag, "pcm recovery failed: %s", snd_strerror(n));
                break;
            }
            continue;
        }
        if (n == 0) continue;

        // 重采样到目标采样率
        resample_s16(raw.data(), hw_rate_, static_cast<size_t>(n),
                     resampled, cfg_.sample_rate);

        // 组装 AudioFrame（深拷贝出 ALSA 缓冲）
        AudioFrame frame;
        const size_t bytes = resampled.size() * sizeof(int16_t);
        frame.data.resize(bytes);
        std::memcpy(frame.data.data(), resampled.data(), bytes);
        frame.sample_rate = cfg_.sample_rate;
        frame.channels    = hw_channels_;
        frame.format_bits = 16;

        // PTS：基于输出样本数 / 目标采样率，叠加起始基准
        const uint64_t dur_ms = (output_samples_ * 1000ULL) / cfg_.sample_rate;
        frame.pts_ms = start_ms_ + dur_ms;
        output_samples_ += resampled.size() / hw_channels_;

        int dropped = ring_->push(frame);
        if (dropped > 0) dropped_.fetch_add(dropped);
        captured_.fetch_add(1);
    }
    LOGI(kTag, "capture_loop exited");
}

bool AlsaCapture::get_frame(AudioFrame& frame, int timeout_ms) {
    if (!ring_) return false;
    return ring_->pop(frame, timeout_ms);
}

#else  // !DMS_HAS_ALSA

// ======================== Stub 实现（无 ALSA 依赖，可编译跑通主流程） ========================
// 当开发机没有 alsa 库时，提供一份"假采集"实现：输出静音帧，方便
// 调试整条音视频流水线的 PTS 对齐、封装逻辑，而不必上板。

AlsaCapture::AlsaCapture() = default;
AlsaCapture::~AlsaCapture() { stop(); }

bool AlsaCapture::open(const AudioConfig& cfg) {
    cfg_ = cfg;
    hw_rate_ = cfg.sample_rate;
    hw_channels_ = cfg.channels;
    hw_format_bytes_ = 2;
    ring_ = std::make_unique<RingBuffer<AudioFrame>>(8);
    LOGW(kTag, "[STUB] ALSA not linked, using silence generator "
               "(target=%uch%u@%uHz)", cfg.channels, cfg.channels, cfg.sample_rate);
    return true;
}

bool AlsaCapture::start() {
    if (running_.exchange(true)) return true;
    start_ms_ = now_ms();
    output_samples_ = 0;
    captured_ = 0;
    dropped_ = 0;
    ring_->reset();
    thread_ = std::thread([this] {
        LOGW(kTag, "[STUB] silence capture thread tid=%u",
             static_cast<unsigned>(::gettid()));
        const uint32_t period = cfg_.period_frames;
        const size_t bytes = period * hw_channels_ * sizeof(int16_t);
        while (running_.load()) {
            AudioFrame frame;
            frame.data.assign(bytes, 0);  // 静音
            frame.sample_rate = cfg_.sample_rate;
            frame.channels = hw_channels_;
            frame.format_bits = 16;
            const uint64_t dur_ms = (output_samples_ * 1000ULL) / cfg_.sample_rate;
            frame.pts_ms = start_ms_ + dur_ms;
            output_samples_ += period;
            int dropped = ring_->push(frame);
            if (dropped > 0) dropped_.fetch_add(dropped);
            captured_.fetch_add(1);
            // 模拟 ALSA 周期等待
            std::this_thread::sleep_for(
                std::chrono::milliseconds(period * 1000 / cfg_.sample_rate));
        }
        LOGW(kTag, "[STUB] silence capture exited");
    });
    return true;
}

void AlsaCapture::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    if (ring_) ring_->stop();
}

void AlsaCapture::resample_s16(const int16_t*, uint32_t, size_t,
                               std::vector<int16_t>&, uint32_t) {}

bool AlsaCapture::get_frame(AudioFrame& frame, int timeout_ms) {
    if (!ring_) return false;
    return ring_->pop(frame, timeout_ms);
}

#endif  // DMS_HAS_ALSA

}  // namespace dms
