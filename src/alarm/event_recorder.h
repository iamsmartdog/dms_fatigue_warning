#ifndef DMS_EVENT_RECORDER_H
#define DMS_EVENT_RECORDER_H

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "utils/media_types.h"

namespace dms {

class Mp4Muxer;

// 事件触发录像管理器
//
// 策略（行车记录仪式事件录像）：
// 1. 维护一个"近实时"滚动缓冲（最近 pre_roll_ms 的码流包），平时不落盘
// 2. 收到 AlarmEvent(level ≥ 阈值) 时：
//    a) 把缓冲里 pre_roll 内的包写入新 mp4
//    b) 继续接收 post_roll_ms 后停止本次录像
// 3. 文件命名：dms_<timestamp>_<level>.mp4
// 4. 同时处理音频包（与视频包一起进 muxer）
class EventRecorder {
public:
    EventRecorder();
    ~EventRecorder();

    struct Config {
        std::string output_dir   = "./records";
        uint32_t    enc_width    = 1280;
        uint32_t    enc_height   = 720;
        uint32_t    enc_fps      = 30;
        uint32_t    audio_sample_rate = 16000;
        uint32_t    audio_channels = 1;
        bool        encrypt      = true;
        uint64_t    pre_roll_ms  = 5000;
        uint64_t    post_roll_ms = 5000;
        AlarmLevel  trigger_level = AlarmLevel::kWarn;
    };
    void set_config(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    // 初始化：创建输出目录（真实环境需 mkdir -p）
    bool init();

    // 喂入编码后的码流包（视频/音频），平时进滚动缓冲，事件触发时落盘
    void feed_packet(const EncodedPacket& pkt);

    // 收到预警事件：触发一次录像
    void on_alarm(const AlarmEvent& ev);

    // 刷新：把进行中的录像收尾（退出时调用）
    void flush();

    uint32_t active_recordings() const { return active_.load(); }
    uint32_t total_recordings()  const { return total_.load(); }

private:
    // 一个进行中的录像会话
    struct Session {
        std::unique_ptr<Mp4Muxer> muxer;
        uint64_t start_ms = 0;
        uint64_t stop_at_ms = 0;
        AlarmLevel level = AlarmLevel::kNone;
        bool stopped = false;
    };

    Config cfg_;
    std::mutex mtx_;

    // 滚动缓冲：保留最近 (pre_roll + 余量) 的包
    std::deque<EncodedPacket> ring_;
    uint64_t ring_first_pts_ = 0;
    uint64_t ring_last_pts_ = 0;
    static constexpr size_t kRingMaxPackets = 600;  // 上限保护

    // 当前进行中的录像会话（一次只允许一个，新事件延续 post_roll）
    std::unique_ptr<Session> session_;

    std::atomic<uint32_t> active_{0};
    std::atomic<uint32_t> total_{0};

    bool inited_ = false;

    static constexpr const char* kTag = "REC";
};

}  // namespace dms

#endif  // DMS_EVENT_RECORDER_H
