#include "encoder/aac_encoder.h"

#include <algorithm>
#include <cstring>

#if defined(DMS_HAS_FDKAAC)
#include <fdk-aac/aacenc_lib.h>
#elif defined(DMS_HAS_FAAC)
#include <faac.h>
#endif

#include "utils/log.h"

namespace dms {

AacEncoder::AacEncoder() = default;

AacEncoder::~AacEncoder() { stop(); }

#if defined(DMS_HAS_FDKAAC)

// ============== 真实 FDK-AAC 实现 ==============

bool AacEncoder::open(const AudioConfig& cfg) {
    cfg_ = cfg;

    HANDLE_AACENCODER h = nullptr;
    if (aacEncOpen(&h, 0, cfg.channels) != AACENC_OK) {
        LOGE(kTag, "aacEncOpen failed");
        return false;
    }

    CHANNEL_MODE mode = cfg.channels == 1 ? MODE_1 : MODE_2;
    if (aacEncoder_SetParam(h, AAC_ENC_SAMPLERATE, cfg.sample_rate) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_CHANNELMODE, mode) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_BITRATE, 64000) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_AUDIOMUX, 1024) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_AOT, AOT_AAC_LC) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_TRANSMUX, TT_MP4_RAW) != AACENC_OK ||
        aacEncoder_SetParam(h, AAC_ENC_AFTERBURNER, 1) != AACENC_OK) {
        LOGE(kTag, "aacEncoder_SetParam failed");
        aacEncClose(&h);
        return false;
    }
    if (aacEncEncode(h, nullptr, nullptr, nullptr, nullptr) != AACENC_OK) {
        LOGE(kTag, "aacEncEncode init failed");
        aacEncClose(&h);
        return false;
    }

    AACENC_InfoStruct info = {};
    aacEncInfo(h, &info);
    input_samples_     = info.frameLength * cfg.channels;
    max_output_bytes_  = info.maxOutBufBytes;
    aac_object_        = 2;  // LC

    handle_ = h;
    pcm_buf_.reserve(input_samples_);

    // AudioSpecificConfig：LC = object 2
    // asc[0] = (5<<3 | (sample_rate_idx >> 1))
    // asc[1] = ((sample_rate_idx & 1)<<7 | (channel<<3))
    static const int sr_idx[16] = {96000,88200,64000,48000,44100,32000,
                                   24000,22050,16000,12000,11025,8000,7350};
    int sr_i = 4;  // 44100 默认
    for (int i = 0; i < 12; ++i) if ((int)cfg.sample_rate == sr_idx[i]) { sr_i = i; break; }
    asc_[0] = static_cast<uint8_t>((aac_object_ << 3) | (sr_i >> 1));
    asc_[1] = static_cast<uint8_t>(((sr_i & 1) << 7) | (cfg.channels << 3));

    ring_ = std::make_unique<RingBuffer<AudioFrame>>(16);
    LOGI(kTag, "FDK-AAC opened: %uch@%uHz frameLength=%u",
         cfg.channels, cfg.sample_rate, info.frameLength);
    return true;
}

bool AacEncoder::encode_block(const int16_t* pcm, size_t frames,
                              uint64_t pts_ms, bool is_first) {
    HANDLE_AACENCODER h = static_cast<HANDLE_AACENCODER>(handle_);
    const int ch = cfg_.channels;
    const int in_size = frames * ch * sizeof(int16_t);

    AACENC_BufDesc in_desc{};
    AACENC_BufDesc out_desc{};
    AACENC_InArgs in_args{};
    AACENC_OutArgs out_args{};

    int in_ids = IN_AUDIO_DATA;
    int in_sizes = in_size;
    void* in_ptrs[1] = { const_cast<int16_t*>(pcm) };
    in_desc.numBufs = 1; in_desc.bufs = in_ptrs;
    in_desc.bufferIdentifiers = &in_ids; in_desc.bufSizes = &in_sizes;
    in_desc.bufElSizes = reinterpret_cast<int*>(nullptr);
    in_args.numInSamples = frames * ch;

    std::vector<uint8_t> out_buf(max_output_bytes_);
    void* out_ptrs[1] = { out_buf.data() };
    int out_ids = OUT_BITSTREAM;
    int out_sizes = out_buf.size();
    out_desc.numBufs = 1; out_desc.bufs = out_ptrs;
    out_desc.bufferIdentifiers = &out_ids; out_desc.bufSizes = &out_sizes;
    out_desc.bufElSizes = reinterpret_cast<int*>(nullptr);

    if (aacEncEncode(h, &in_desc, &out_desc, &in_args, &out_args) != AACENC_OK) {
        LOGE(kTag, "aacEncEncode failed");
        return false;
    }

    EncodedPacket pkt;
    pkt.type = MediaType::kAudio;
    pkt.pts_ms = pts_ms;
    pkt.dts_ms = pts_ms;
    pkt.data.assign(out_buf.data(), out_buf.data() + out_args.numOutBytes);
    if (on_encoded_) on_encoded_(pkt);
    encoded_.fetch_add(1);
    return true;
}

void AacEncoder::get_audio_specific_config(uint8_t asc[2]) const {
    asc[0] = asc_[0]; asc[1] = asc_[1];
}

#elif defined(DMS_HAS_FAAC)

// ============== faac 实现 ==============
bool AacEncoder::open(const AudioConfig& cfg) {
    cfg_ = cfg;
    faacEncHandle h = faacEncOpen(cfg.sample_rate, cfg.channels,
                                  &input_samples_, &max_output_bytes_);
    if (!h) { LOGE(kTag, "faacEncOpen failed"); return false; }
    faacEncConfigurationPtr fc = faacEncGetCurrentConfiguration(h);
    fc->inputFormat = FAAC_INPUT_16BIT;
    fc->outputFormat = 0;  // raw, 不带 ADTS（MP4 用 raw）
    fc->mpegVersion = MPEG4;
    fc->aacObjectType = LOW;
    faacEncSetConfiguration(h, fc);
    handle_ = h;
    pcm_buf_.reserve(input_samples_);
    aac_object_ = 2;
    ring_ = std::make_unique<RingBuffer<AudioFrame>>(16);
    LOGI(kTag, "faac opened: %uch@%uHz input_samples=%lu",
         cfg.channels, cfg.sample_rate, input_samples_);
    return true;
}

bool AacEncoder::encode_block(const int16_t* pcm, size_t frames,
                              uint64_t pts_ms, bool is_first) {
    faacEncHandle h = static_cast<faacEncHandle>(handle_);
    std::vector<uint8_t> out(max_output_bytes_);
    int n = faacEncEncode(h, const_cast<int32_t*>(reinterpret_cast<const int32_t*>(pcm)),
                          input_samples_ * sizeof(int16_t),
                          out.data(), out.size());
    if (n < 0) { LOGE(kTag, "faacEncEncode failed"); return false; }
    if (n == 0) return true;
    EncodedPacket pkt;
    pkt.type = MediaType::kAudio;
    pkt.pts_ms = pts_ms; pkt.dts_ms = pts_ms;
    pkt.data.assign(out.data(), out.data() + n);
    if (on_encoded_) on_encoded_(pkt);
    encoded_.fetch_add(1);
    return true;
}

void AacEncoder::get_audio_specific_config(uint8_t asc[2]) const {
    asc[0] = asc_[0]; asc[1] = asc_[1];
}

#else

// ============== Stub ==============
bool AacEncoder::open(const AudioConfig& cfg) {
    cfg_ = cfg;
    input_samples_ = 1024 * cfg.channels;
    max_output_bytes_ = 2048;
    pcm_buf_.reserve(input_samples_);
    ring_ = std::make_unique<RingBuffer<AudioFrame>>(16);
    LOGW(kTag, "[STUB] AAC encoder (no FDK/FAAC linked) opened");
    return true;
}

bool AacEncoder::encode_block(const int16_t* pcm, size_t frames,
                              uint64_t pts_ms, bool is_first) {
    // 模拟 AAC 帧占位
    EncodedPacket pkt;
    pkt.type = MediaType::kAudio;
    pkt.pts_ms = pts_ms; pkt.dts_ms = pts_ms;
    pkt.data.assign(8, 0);  // 8 字节占位
    if (on_encoded_) on_encoded_(pkt);
    encoded_.fetch_add(1);
    return true;
}

void AacEncoder::get_audio_specific_config(uint8_t asc[2]) const {
    asc[0] = asc_[0]; asc[1] = asc_[1];
}

#endif

// ============== 公共：缓冲、线程 ==============

int AacEncoder::push_frame(const AudioFrame& frame) {
    if (!ring_) return -1;
    int dropped = ring_->push(frame);
    return dropped;
}

bool AacEncoder::start() {
    if (running_.exchange(true)) return true;
    if (!ring_) { running_ = false; LOGE(kTag, "start failed: ring not initialized"); return false; }
    encoded_ = 0;
    pcm_buf_.clear();
    buf_samples_filled_ = 0;
    first_block_ = true;
    if (ring_) ring_->reset();
    thread_ = std::thread(&AacEncoder::encode_loop, this);
    LOGI(kTag, "aac encode thread started");
    return true;
}

void AacEncoder::stop() {
    if (!running_.exchange(false)) return;
    if (ring_) ring_->stop();
    if (thread_.joinable()) thread_.join();
#if defined(DMS_HAS_FDKAAC)
    if (handle_) { aacEncClose(reinterpret_cast<HANDLE_AACENCODER*>(&handle_)); handle_ = nullptr; }
#elif defined(DMS_HAS_FAAC)
    if (handle_) { faacEncClose(handle_); handle_ = nullptr; }
#endif
    LOGI(kTag, "aac closed, encoded=%lu", encoded_.load());
}

void AacEncoder::encode_loop() {
    LOGI(kTag, "encode_loop tid=%u", static_cast<unsigned>(::gettid()));
    const size_t frame_samples = input_samples_;  // AAC 一帧所需样本数（含通道）
    const uint32_t ch = cfg_.channels;

    while (running_.load()) {
        AudioFrame af;
        if (!ring_->pop(af, 100)) continue;

        const int16_t* src = reinterpret_cast<const int16_t*>(af.data.data());
        size_t src_samples = af.data.size() / sizeof(int16_t);
        size_t src_consumed = 0;

        // 把 PCM 累积到 pcm_buf_，凑满一帧 AAC 就编码
        while (src_consumed < src_samples) {
            size_t need = frame_samples - buf_samples_filled_;
            size_t copy = std::min(need, src_samples - src_consumed);
            pcm_buf_.insert(pcm_buf_.end(),
                            src + src_consumed, src + src_consumed + copy);
            buf_samples_filled_ += copy;
            src_consumed += copy;

            if (buf_samples_filled_ >= frame_samples) {
                // 这一帧 AAC 的 PTS：基于该块起始样本对应的 ms
                uint64_t pts = buf_pts_base_ms_;
                if (first_block_) {
                    buf_pts_base_ms_ = af.pts_ms;
                    pts = buf_pts_base_ms_;
                    first_block_ = false;
                }
                encode_block(pcm_buf_.data(), frame_samples / ch, pts, false);
                // 推进 PTS：1024 samples / sample_rate 秒
                buf_pts_base_ms_ += (1024ULL * 1000) / cfg_.sample_rate;
                pcm_buf_.clear();
                buf_samples_filled_ = 0;
            } else {
                // 当前 input frame 不够一帧，保留在 buf，等下一帧
                if (first_block_) {
                    buf_pts_base_ms_ = af.pts_ms;
                    first_block_ = false;
                }
            }
        }
    }
    LOGI(kTag, "encode_loop exited");
}

}  // namespace dms
