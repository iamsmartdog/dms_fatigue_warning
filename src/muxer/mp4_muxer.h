#ifndef DMS_MP4_MUXER_H
#define DMS_MP4_MUXER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "utils/media_types.h"

namespace dms {

// MP4 封装配置
struct MuxerConfig {
    std::string output_path;          // 输出文件
    uint32_t video_width   = 1280;
    uint32_t video_height  = 720;
    uint32_t video_fps     = 30;
    uint32_t video_timescale = 3000;  // 视频轨时间基（3000 = 整除 30/60fps）
    uint32_t audio_sample_rate = 16000;
    uint32_t audio_channels    = 1;
    uint32_t audio_timescale = 16000;
    bool encrypt = false;             // 是否对 sample 做 AES 加密
    // 每文件最大时长（秒），达到自动切下一个文件（事件录像分片）
    uint32_t segment_sec = 0;         // 0 = 不分片
};

// MP4 封装器（音视频 PTS 对齐 + 可选加密）
//
// 底层实现要点：
// 1. 用轻量级 mp4 muxer（如 libmp4 / 自写 ISO BMFF muxer）写出 ftyp/mdat/moov
// 2. 音视频 PTS 统一到毫秒域，按轨写入时换算为该轨 timescale 下的 DTS
// 3. 视频按到达顺序写入，音频按 PTS 排序插入，moov 写在文件末尾
// 4. encrypt=true 时，每个 sample 在写入 mdat 前过 AesCrypto::instance().encrypt
//
// 这里用宏 DMS_HAS_MP4V2 包裹 mp4v2 库调用；stub 实现写 .bin 调试用
class Mp4Muxer {
public:
    Mp4Muxer();
    ~Mp4Muxer();

    Mp4Muxer(const Mp4Muxer&) = delete;
    Mp4Muxer& operator=(const Mp4Muxer&) = delete;

    // 打开文件，写 ftyp + 初始化轨道
    bool open(const MuxerConfig& cfg);

    // 写入一个编码包（视频或音频）
    bool write_packet(const EncodedPacket& pkt);

    // 关闭：flush 剩余、写 moov、关闭文件
    void close();

    bool is_open() const { return opened_; }
    uint64_t video_samples() const { return v_samples_; }
    uint64_t audio_samples() const { return a_samples_; }

    // 诊断：最近一次音视频 PTS 差（ms），用于评估同步质量
    int64_t av_pts_diff_ms() const { return last_pts_diff_ms_; }

private:
    // 计算每个 sample 的持续时间（基于相邻 PTS 差）
    uint32_t video_duration_ms(const EncodedPacket& pkt);
    uint32_t audio_duration_ms(const EncodedPacket& pkt);

    MuxerConfig cfg_;
    bool opened_ = false;

    // PTS 跟踪（音视频对齐用）
    uint64_t first_v_pts_ = 0;
    uint64_t first_a_pts_ = 0;
    uint64_t prev_v_pts_ = 0;
    uint64_t prev_a_pts_ = 0;
    bool has_v_ = false;
    bool has_a_ = false;
    int64_t  last_pts_diff_ms_ = 0;
    uint64_t v_samples_ = 0;
    uint64_t a_samples_ = 0;
    uint64_t seg_start_ms_ = 0;

    // mp4v2 句柄（仅 DMS_HAS_MP4V2）
    void* mp4_handle_ = nullptr;
    int   video_track_ = -1;
    int   audio_track_ = -1;

    // Stub：原始文件句柄
    void* stub_fp_ = nullptr;
    std::mutex mtx_;

    static constexpr const char* kTag = "MUX";
};

}  // namespace dms

#endif  // DMS_MP4_MUXER_H
