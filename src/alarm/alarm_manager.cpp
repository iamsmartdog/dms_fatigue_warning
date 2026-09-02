#include "alarm/alarm_manager.h"

#include "utils/log.h"

namespace dms {

AlarmManager::AlarmManager() = default;
AlarmManager::~AlarmManager() = default;

void AlarmManager::speak(const std::string& text) {
    // 真实实现：调用 espeak / piper TTS + aplay，或写 /dev/ttyS 给 MCU 喇叭
    // 此处仅打日志（避免依赖）。若 enable_tts=false 也走这里。
    if (cfg_.enable_tts) {
        LOGI(kTag, "[TTS] %s", text.c_str());
    }
}

bool AlarmManager::handle(const AlarmEvent& ev) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (ev.level == AlarmLevel::kNone) return false;

    uint64_t now = ev.pts_ms;
    uint64_t* last = nullptr;
    uint64_t  cooldown = 0;
    const char* tts_text = "";
    std::atomic<uint64_t>* counter = nullptr;

    switch (ev.level) {
        case AlarmLevel::kInfo:
            last = &last_info_ms_; cooldown = cfg_.info_cooldown_ms;
            counter = &info_; tts_text = "请注意保持清醒";
            break;
        case AlarmLevel::kWarn:
            last = &last_warn_ms_; cooldown = cfg_.warn_cooldown_ms;
            counter = &warn_; tts_text = "检测到疲劳，请就近停车休息";
            break;
        case AlarmLevel::kDanger:
            last = &last_danger_ms_; cooldown = cfg_.danger_cooldown_ms;
            counter = &danger_; tts_text = "严重疲劳，请立即停车！";
            break;
        default:
            return false;
    }

    // 冷却判定（首次触发 last=0 必触发）
    if (*last != 0 && (now - *last) < cooldown) {
        return false;  // 同级别冷却中，不重复触发
    }
    *last = now;
    counter->fetch_add(1);

    // 日志 + TTS
    LOGI(kTag, "[%s] %s (pts=%lu metric=%.2f)",
         alarm_level_str(ev.level), ev.reason.c_str(),
         (unsigned long)ev.pts_ms, ev.metric);
    speak(tts_text);

    // 蜂鸣器（GPIO / 串口）
    if (cfg_.enable_buzzer && ev.level == AlarmLevel::kDanger) {
        LOGI(kTag, "[BUZZER] ON (danger)");
    }

    // 回调：通知 UI / 触发事件录像
    if (cb_) {
        // 若达到录像阈值，标记需要录像
        AlarmEvent out = ev;
        bool need_record =
            static_cast<int>(ev.level) >= static_cast<int>(cfg_.record_trigger_level);
        // 复用 pre_roll/post_roll 字段通知录像系统
        if (need_record) {
            LOGI(kTag, "trigger event record (pre=%lums post=%lums)",
                 (unsigned long)ev.pre_roll_ms, (unsigned long)ev.post_roll_ms);
        }
        cb_(out);
    }
    return true;
}

}  // namespace dms
