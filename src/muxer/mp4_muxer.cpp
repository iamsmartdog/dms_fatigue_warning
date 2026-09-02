#include "muxer/mp4_muxer.h"

#include <algorithm>
#include <cstring>

#ifdef DMS_HAS_MP4V2
#include <mp4v2/mp4v2.h>
#endif

#include "crypto/aes_crypto.h"
#include "utils/log.h"

namespace dms {

Mp4Muxer::Mp4Muxer() = default;
Mp4Muxer::~Mp4Muxer() { close(); }

#ifdef DMS_HAS_MP4V2

// ==================== mp4v2 实现 ====================
bool Mp4Muxer::open(const MuxerConfig& cfg) {
    cfg_ = cfg;

    MP4FileHandle h = MP4CreateEx(cfg_.output_path.c_str(), 0, 1, 1, 0, 0, 0, 0);
    if (!h) { LOGE(kTag, "MP4Create %s failed", cfg_.output_path.c_str()); return false; }
    mp4_handle_ = h;
    MP4SetTimeScale(h, 1000);  // 全局 ms 基准

    // 视频：HEVC
    video_track_ = MP4AddHvcVideoTrack(h, cfg_.video_timescale,
                                       cfg_.video_timescale / cfg_.video_fps,
                                       cfg_.video_width, cfg_.video_height);
    if (video_track_ == MP4_INVALID_TRACK_ID) {
        LOGE(kTag, "add HEVC track failed"); return false;
    }
    // 音频：AAC
    audio_track_ = MP4AddAudioTrack(h, cfg_.audio_timescale,
                                    cfg_.audio_timescale / 50,  // AAC 1024/48000 近似
                                    MP4_MPEG4_AUDIO_TYPE);
    if (audio_track_ == MP4_INVALID_TRACK_ID) {
        LOGE(kTag, "add AAC track failed"); return false;
    }

    opened_ = true;
    seg_start_ms_ = now_ms();
    LOGI(kTag, "MP4 opened: %s (hevc + aac) encrypt=%d",
         cfg_.output_path.c_str(), (int)cfg_.encrypt);
    return true;
}

uint32_t Mp4Muxer::video_duration_ms(const EncodedPacket& pkt) {
    if (!has_v_) return cfg_.video_timescale / cfg_.video_fps;
    uint64_t d = (pkt.pts_ms > prev_v_pts_) ? (pkt.pts_ms - prev_v_pts_) : 0;
    return static_cast<uint32_t>(d * cfg_.video_timescale / 1000);
}
uint32_t Mp4Muxer::audio_duration_ms(const EncodedPacket& pkt) {
    if (!has_a_) return cfg_.audio_timescale / 50;
    uint64_t d = (pkt.pts_ms > prev_a_pts_) ? (pkt.pts_ms - prev_a_pts_) : 0;
    return static_cast<uint32_t>(d * cfg_.audio_timescale / 1000);
}

bool Mp4Muxer::write_packet(const EncodedPacket& pkt) {
    if (!opened_) return false;
    std::lock_guard<std::mutex> lock(mtx_);

    MP4FileHandle h = static_cast<MP4FileHandle>(mp4_handle_);
    const uint8_t* payload = pkt.data.data();
    std::vector<uint8_t> cipher;
    if (cfg_.encrypt) {
        AesCrypto::instance().encrypt(pkt.data, cipher);
        payload = cipher.data();
    }

    if (pkt.type == MediaType::kVideo) {
        if (!has_v_) { first_v_pts_ = pkt.pts_ms; has_v_ = true; }
        uint64_t dts = (pkt.pts_ms - first_v_pts_) * cfg_.video_timescale / 1000;
        MP4WriteSample(h, video_track_, payload, pkt.data.size(),
                       video_duration_ms(pkt), dts,
                       pkt.keyframe ? MP4_SAMPLE_FLAG_SYNC : 0);
        prev_v_pts_ = pkt.pts_ms; ++v_samples_;
        if (has_a_) last_pts_diff_ms_ = (int64_t)pkt.pts_ms - (int64_t)prev_a_pts_;
    } else {
        if (!has_a_) { first_a_pts_ = pkt.pts_ms; has_a_ = true; }
        uint64_t dts = (pkt.pts_ms - first_a_pts_) * cfg_.audio_timescale / 1000;
        MP4WriteSample(h, audio_track_, payload, pkt.data.size(),
                       audio_duration_ms(pkt), dts, MP4_SAMPLE_FLAG_SYNC);
        prev_a_pts_ = pkt.pts_ms; ++a_samples_;
        if (has_v_) last_pts_diff_ms_ = (int64_t)pkt.pts_ms - (int64_t)prev_v_pts_;
    }

    // 分片
    if (cfg_.segment_sec > 0) {
        uint64_t elapsed = (pkt.pts_ms - (has_v_ ? first_v_pts_ : first_a_pts_)) / 1000;
        if (elapsed >= cfg_.segment_sec) {
            LOGI(kTag, "segment threshold reached, finalize + reopen");
            // 简化：触发回调由上层重开新 muxer
        }
    }
    return true;
}

void Mp4Muxer::close() {
    if (!opened_) return;
    opened_ = false;
    if (mp4_handle_) {
        MP4Close(static_cast<MP4FileHandle>(mp4_handle_), 0);
        mp4_handle_ = nullptr;
    }
    LOGI(kTag, "MP4 closed v=%lu a=%lu pts_diff=%ldms",
         v_samples_, a_samples_, last_pts_diff_ms_);
}

#else

// ==================== Stub：写裸码流 .bin，便于开发机调试 ====================
bool Mp4Muxer::open(const MuxerConfig& cfg) {
    cfg_ = cfg;
    std::string path = cfg_.output_path;
    if (path.empty()) path = "dms_dump.bin";
    stub_fp_ = std::fopen(path.c_str(), "wb");
    if (!stub_fp_) { LOGE(kTag, "fopen %s failed", path.c_str()); return false; }
    // 写一个简易文件头：标识 + 视频/音频参数
    struct { char magic[4] = {'D','M','S','1'}; uint32_t w, h, fps, sr, ch; } hdr;
    hdr.w = cfg_.video_width; hdr.h = cfg_.video_height; hdr.fps = cfg_.video_fps;
    hdr.sr = cfg_.audio_sample_rate; hdr.ch = cfg_.audio_channels;
    std::fwrite(&hdr, sizeof(hdr), 1, static_cast<FILE*>(stub_fp_));
    opened_ = true;
    LOGW(kTag, "[STUB] mp4v2 not linked, writing raw stream -> %s", path.c_str());
    return true;
}

uint32_t Mp4Muxer::video_duration_ms(const EncodedPacket&) { return 0; }
uint32_t Mp4Muxer::audio_duration_ms(const EncodedPacket&) { return 0; }

bool Mp4Muxer::write_packet(const EncodedPacket& pkt) {
    if (!opened_ || !stub_fp_) return false;
    std::lock_guard<std::mutex> lock(mtx_);

    const uint8_t* payload = pkt.data.data();
    std::vector<uint8_t> cipher;
    if (cfg_.encrypt) {
        AesCrypto::instance().encrypt(pkt.data, cipher);
        payload = cipher.data();
    }

    // 简易 record: [type:1][key:1][pts:8][len:4][data:len]
    uint8_t  type = (pkt.type == MediaType::kVideo) ? 0 : 1;
    uint8_t  key  = pkt.keyframe ? 1 : 0;
    uint64_t pts  = pkt.pts_ms;
    uint32_t len  = static_cast<uint32_t>(pkt.data.size());
    std::fwrite(&type, 1, 1, static_cast<FILE*>(stub_fp_));
    std::fwrite(&key,  1, 1, static_cast<FILE*>(stub_fp_));
    std::fwrite(&pts,  8, 1, static_cast<FILE*>(stub_fp_));
    std::fwrite(&len,  4, 1, static_cast<FILE*>(stub_fp_));
    std::fwrite(payload, 1, len, static_cast<FILE*>(stub_fp_));

    if (pkt.type == MediaType::kVideo) {
        if (!has_v_) { first_v_pts_ = pkt.pts_ms; has_v_ = true; }
        prev_v_pts_ = pkt.pts_ms; ++v_samples_;
        if (has_a_) last_pts_diff_ms_ = (int64_t)pkt.pts_ms - (int64_t)prev_a_pts_;
    } else {
        if (!has_a_) { first_a_pts_ = pkt.pts_ms; has_a_ = true; }
        prev_a_pts_ = pkt.pts_ms; ++a_samples_;
        if (has_v_) last_pts_diff_ms_ = (int64_t)pkt.pts_ms - (int64_t)prev_v_pts_;
    }
    return true;
}

void Mp4Muxer::close() {
    if (!opened_) return;
    opened_ = false;
    if (stub_fp_) { std::fclose(static_cast<FILE*>(stub_fp_)); stub_fp_ = nullptr; }
    LOGW(kTag, "[STUB] closed v=%lu a=%lu pts_diff=%ldms",
         v_samples_, a_samples_, last_pts_diff_ms_);
}

#endif  // DMS_HAS_MP4V2

}  // namespace dms
