#include "alarm/fatigue_analyzer.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "utils/log.h"

namespace dms {

// ============================================================
// 内置策略实现
// ============================================================

// 1) PERCLOS 策略：长时间滑窗内闭眼占比
class PerclosStrategy : public IFatigueStrategy {
public:
    explicit PerclosStrategy(const FatigueAnalyzer::Config* c) : cfg_(c) {}
    const char* name() const override { return "PERCLOS"; }
    AlarmLevel evaluate(const FatigueContext& ctx, std::string& reason) override {
        if (!ctx.calibrated) return AlarmLevel::kNone;  // 校准期不触发
        if (ctx.perclos >= cfg_->perclos_danger_thresh) {
            reason = "PERCLOS " + std::to_string(ctx.perclos * 100).substr(0, 5) +
                     "% >= " + std::to_string((int)(cfg_->perclos_danger_thresh * 100)) + "%";
            return AlarmLevel::kDanger;
        }
        if (ctx.perclos >= cfg_->perclos_warn_thresh) {
            reason = "PERCLOS " + std::to_string(ctx.perclos * 100).substr(0, 5) +
                     "% >= " + std::to_string((int)(cfg_->perclos_warn_thresh * 100)) + "%";
            return AlarmLevel::kWarn;
        }
        return AlarmLevel::kNone;
    }
private:
    const FatigueAnalyzer::Config* cfg_;
};

// 2) 闭眼时长策略：单次持续闭眼
class EyeClosureStrategy : public IFatigueStrategy {
public:
    explicit EyeClosureStrategy(const FatigueAnalyzer::Config* c) : cfg_(c) {}
    const char* name() const override { return "EyeClosure"; }
    AlarmLevel evaluate(const FatigueContext& ctx, std::string& reason) override {
        if (!ctx.calibrated) return AlarmLevel::kNone;  // 校准期不触发
        if (!ctx.eye_closed) return AlarmLevel::kNone;
        uint64_t dur = ctx.cur_pts - ctx.eye_close_start_ms;
        if (dur >= cfg_->eye_close_danger_ms) {
            reason = "Micro-sleep " + std::to_string(dur / 1000) + "s";
            return AlarmLevel::kDanger;
        }
        if (dur >= cfg_->eye_close_warn_ms) {
            reason = "Eye closed " + std::to_string(dur / 1000) + "s";
            return AlarmLevel::kWarn;
        }
        return AlarmLevel::kNone;
    }
private:
    const FatigueAnalyzer::Config* cfg_;
};

// 3) 打哈欠策略
class YawnStrategy : public IFatigueStrategy {
public:
    explicit YawnStrategy(const FatigueAnalyzer::Config* c) : cfg_(c) {}
    const char* name() const override { return "Yawn"; }
    AlarmLevel evaluate(const FatigueContext& ctx, std::string& reason) override {
        if (!ctx.mouth_open) return AlarmLevel::kNone;
        uint64_t dur = ctx.cur_pts - ctx.mouth_open_start_ms;
        if (dur >= cfg_->yawn_warn_ms) {
            reason = "Yawning " + std::to_string(dur / 1000) + "s";
            return AlarmLevel::kWarn;
        }
        return AlarmLevel::kNone;
    }
private:
    const FatigueAnalyzer::Config* cfg_;
};

// 4) 头部低垂策略
class HeadDropStrategy : public IFatigueStrategy {
public:
    explicit HeadDropStrategy(const FatigueAnalyzer::Config* c) : cfg_(c) {}
    const char* name() const override { return "HeadDrop"; }
    AlarmLevel evaluate(const FatigueContext& ctx, std::string& reason) override {
        if (!ctx.head_dropping) return AlarmLevel::kNone;
        uint64_t dur = ctx.cur_pts - ctx.head_drop_start_ms;
        if (dur >= cfg_->head_drop_warn_ms) {
            reason = "Head dropping " + std::to_string(dur / 1000) + "s";
            return AlarmLevel::kWarn;
        }
        return AlarmLevel::kNone;
    }
private:
    const FatigueAnalyzer::Config* cfg_;
};

// ============================================================
// 模板方法实现
// ============================================================

FatigueAnalyzer::FatigueAnalyzer() {
    ctx_.window_ms = 60000;
    // 默认挂载四个策略，可在 main 中 set_config 关闭 / add_strategy 替换
    if (true) strategies_.push_back(std::make_unique<PerclosStrategy>(&cfg_));
    if (true) strategies_.push_back(std::make_unique<EyeClosureStrategy>(&cfg_));
    if (true) strategies_.push_back(std::make_unique<YawnStrategy>(&cfg_));
    if (true) strategies_.push_back(std::make_unique<HeadDropStrategy>(&cfg_));
}

FatigueAnalyzer::~FatigueAnalyzer() = default;

void FatigueAnalyzer::add_strategy(std::unique_ptr<IFatigueStrategy> s) {
    strategies_.push_back(std::move(s));
}

// Step1
void FatigueAnalyzer::extract_features(const DetectionResult& det) {
    ctx_.cur_pts = det.pts_ms;
    if (det.faces.empty()) {
        // 无人脸：保持上一帧 EAR（不更新），避免人短暂离开画面时误报闭眼
        // 低于动态阈值时拉回略高于阈值的睁眼值，避免误触发闭眼计时
        if (ctx_.cur_ear < ctx_.cur_ear_thresh) {
            ctx_.cur_ear = ctx_.cur_ear_thresh + 0.05f;
        }
        return;
    }
    // 取置信度最高的人脸
    const FaceBox& f = det.faces.front();
    // EAR：左右平均
    ctx_.cur_ear = (f.ear_left + f.ear_right) / 2.0f;
    ctx_.cur_mar = f.mar;
    ctx_.cur_pitch = f.head_pitch;

    // ===== 自校准：动态 EAR 阈值（持续适应，不锁死）=====
    if (!ctx_.calibrated) {
        // 初始校准期：累积 EAR
        calib_ears_.push_back(ctx_.cur_ear);
        if (calib_ears_.size() >= kCalibFrames) {
            // 取中位数作 baseline（抗眨眼/噪声）
            std::vector<float> sorted(calib_ears_.begin(), calib_ears_.end());
            std::sort(sorted.begin(), sorted.end());
            ear_baseline_ = sorted[sorted.size() / 2];
            ctx_.cur_ear_thresh = ear_baseline_ - kEarOffset;
            ctx_.calibrated = true;
            LOGI("FATIGUE", "EAR calibrated: baseline=%.3f thresh=%.3f",
                 ear_baseline_, ctx_.cur_ear_thresh);
        }
    } else {
        // 运行期：EMA 平滑（减少瞬间波动）
        // 平滑因子：0.8 表示 80% 历史 + 20% 当前
        ctx_.cur_ear = 0.8f * ctx_.cur_ear_smoothed + 0.2f * ctx_.cur_ear;
        ctx_.cur_ear_smoothed = ctx_.cur_ear;
        
        // baseline 锁死：初始校准后不再刷新（完全避免抖动）
        // 如果需要长期适应，改为极慢刷新（比如 3000 帧而不是 600）
        // 目前保持锁死以消除延时和抖动
    }
}

// Step2
void FatigueAnalyzer::accumulate_timeline() {
    if (ctx_.first_pts == 0) {
        ctx_.first_pts = ctx_.cur_pts;
        blink_window_start_ms_ = ctx_.cur_pts;
    }
    ctx_.last_pts = ctx_.cur_pts;

    // 滑窗 EAR 历史
    ctx_.ear_history.push_back(ctx_.cur_ear);
    while (!ctx_.ear_history.empty() &&
           ctx_.ear_history.size() > 30 * (ctx_.window_ms / 1000)) {
        ctx_.ear_history.pop_front();
    }

    // 闭眼计时 + 眨眼计数（闭→开 算一次眨眼）
    bool now_closed = ctx_.cur_ear < ctx_.cur_ear_thresh;
    
    // 防抖 + 滞后：使用两个不同的阈值避免睁眼状态抖动
    // 闭眼阈值：cur_ear_thresh
    // 睡着状态已经锁定时，需要 cur_ear > (cur_ear_thresh + 0.15) 才能唤醒（增加滞后）
    if (ctx_.eye_closed) {
        // 已经闭眼，需要更高的值才能睁眼
        now_closed = ctx_.cur_ear < (ctx_.cur_ear_thresh + 0.15f);
    }
    
    // 防止单帧噪声：连续 5 帧（而不是 3 帧）才切换状态（加强防抖）
    if (now_closed) {
        ++eye_closed_frame_count_;
        eye_open_frame_count_ = 0;
    } else {
        ++eye_open_frame_count_;
        eye_closed_frame_count_ = 0;
    }
    
    const int kStateChangeThreshold = 5;  // 从 3 改为 5
    if (eye_closed_frame_count_ >= kStateChangeThreshold && !ctx_.eye_closed) {
        ctx_.eye_closed = true;
        ctx_.eye_close_start_ms = ctx_.cur_pts;
    } else if (eye_open_frame_count_ >= kStateChangeThreshold && ctx_.eye_closed) {
        // 睁开
        uint64_t dur = ctx_.cur_pts - ctx_.eye_close_start_ms;
        if (dur > ctx_.max_eye_closure_ms) ctx_.max_eye_closure_ms = dur;
        // 短闭眼（< 0.5s）算正常眨眼
        if (dur < 500) {
            ++blink_count_;
        }
        ctx_.eye_closed = false;
    }

    // 嘴部
    bool now_mouth = ctx_.cur_mar > cfg_.mar_yawn_thresh;
    if (now_mouth && !ctx_.mouth_open) {
        ctx_.mouth_open = true;
        ctx_.mouth_open_start_ms = ctx_.cur_pts;
    } else if (!now_mouth && ctx_.mouth_open) {
        ctx_.mouth_open = false;
    }

    // 头部低垂
    bool now_drop = ctx_.cur_pitch < cfg_.pitch_drop_thresh;
    if (now_drop && !ctx_.head_dropping) {
        ctx_.head_dropping = true;
        ctx_.head_drop_start_ms = ctx_.cur_pts;
    } else if (!now_drop && ctx_.head_dropping) {
        ctx_.head_dropping = false;
    }
}

// Step3
void FatigueAnalyzer::compute_metrics() {
    // PERCLOS：ear_history 中 < 阈值的比例
    if (ctx_.ear_history.empty()) {
        ctx_.perclos = 0;
    } else {
        size_t closed = 0;
        for (float v : ctx_.ear_history) {
            if (v < ctx_.cur_ear_thresh) ++closed;
        }
        ctx_.perclos = static_cast<float>(closed) / ctx_.ear_history.size();
    }
    // 眨眼频率（次/分钟）
    uint64_t elapsed = ctx_.cur_pts - blink_window_start_ms_;
    if (elapsed > 0) {
        ctx_.blink_rate = (blink_count_ * 60000.0f) / static_cast<float>(elapsed);
    }
}

// Step4
AlarmLevel FatigueAnalyzer::aggregate_decisions(std::string& reason, float& metric) {
    AlarmLevel max_lv = AlarmLevel::kNone;
    reason = "OK";
    metric = ctx_.perclos;
    for (auto& s : strategies_) {
        std::string r;
        AlarmLevel lv = s->evaluate(ctx_, r);
        if (static_cast<int>(lv) > static_cast<int>(max_lv)) {
            max_lv = lv;
            reason = std::string(s->name()) + ": " + r;
            metric = ctx_.perclos;
        }
    }
    return max_lv;
}

// 模板方法主入口
AlarmEvent FatigueAnalyzer::process(const DetectionResult& det) {
    extract_features(det);
    accumulate_timeline();
    compute_metrics();

    std::string reason;
    float metric = 0;
    AlarmLevel lv = aggregate_decisions(reason, metric);

    AlarmEvent ev;
    ev.level    = lv;
    ev.reason   = std::move(reason);
    ev.pts_ms   = det.pts_ms;
    ev.metric   = metric;
    ev.pre_roll_ms  = 5000;
    ev.post_roll_ms = 5000;
    return ev;
}

}  // namespace dms
