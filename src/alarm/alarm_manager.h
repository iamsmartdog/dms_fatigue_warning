#ifndef DMS_ALARM_MANAGER_H
#define DMS_ALARM_MANAGER_H

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

#include "utils/media_types.h"

namespace dms {

// 预警输出动作回调（向上层 / UI / 硬件）
//   level       触发级别
//   reason      原因描述
//   pts_ms      触发时刻
//   force_record 是否要求启动事件录像
using AlarmCallback = std::function<void(const AlarmEvent&)>;

// 预警管理器
//
// 职责：
// 1. 接收 FatigueAnalyzer 的 AlarmEvent，做分级抑制（避免相邻帧反复触发）
// 2. 不同级别对应不同输出（INFO: 提示音 / WARN: TTS + 事件录像 / DANGER: 持续报警）
// 3. 维护冷却时间（同级别 N 秒内不重复触发）
// 4. 触发"事件录像"：通知上层 Encoder/Muxer 启动一个录像片段（pre+post roll）
class AlarmManager {
public:
    AlarmManager();
    ~AlarmManager();

    struct Config {
        uint64_t info_cooldown_ms   = 10000;  // INFO 级别冷却 10s
        uint64_t warn_cooldown_ms   = 8000;
        uint64_t danger_cooldown_ms = 5000;
        bool enable_tts = true;
        bool enable_buzzer = true;
        // 事件触发录像：达到 WARN 及以上才触发
        AlarmLevel record_trigger_level = AlarmLevel::kWarn;
    };
    void set_config(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    // 注册回调（UI/硬件/录像触发）
    void set_callback(AlarmCallback cb) { cb_ = std::move(cb); }

    // 输入一帧的判定结果，返回是否实际触发了输出（考虑冷却后）
    bool handle(const AlarmEvent& ev);

    // TTS 播报（默认实现仅打日志，子类/回调可重写）
    virtual void speak(const std::string& text);

    // 诊断统计
    uint64_t info_count()   const { return info_.load(); }
    uint64_t warn_count()   const { return warn_.load(); }
    uint64_t danger_count() const { return danger_.load(); }

private:
    Config cfg_;
    AlarmCallback cb_;
    std::mutex mtx_;

    // 各级别最近一次触发时间
    uint64_t last_info_ms_   = 0;
    uint64_t last_warn_ms_   = 0;
    uint64_t last_danger_ms_ = 0;

    std::atomic<uint64_t> info_{0};
    std::atomic<uint64_t> warn_{0};
    std::atomic<uint64_t> danger_{0};

    static constexpr const char* kTag = "ALARM";
};

}  // namespace dms

#endif  // DMS_ALARM_MANAGER_H
