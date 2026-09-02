#ifndef DMS_FATIGUE_ANALYZER_H
#define DMS_FATIGUE_ANALYZER_H

#include <cstdint>
#include <deque>
#include <memory>

#include "utils/media_types.h"

namespace dms {

// 疲劳分析上下文：传递给各判定策略
struct FatigueContext {
    // 滑窗内每帧的 EAR（取 L/R 较小值）
    std::deque<float> ear_history;
    // 滑窗时间（ms）
    uint64_t window_ms = 60000;       // 默认 60s
    uint64_t first_pts = 0;
    uint64_t last_pts  = 0;

    // PERCLOS：滑窗内"眼睑闭合 > 阈值"的时间占比
    float perclos = 0.0f;
    // 平均眨眼频率（次/分钟）
    float blink_rate = 0.0f;
    // 最长连续闭眼时长（ms）
    uint64_t max_eye_closure_ms = 0;
    // 当前是否闭眼 + 闭眼起始时刻
    bool     eye_closed = false;
    uint64_t eye_close_start_ms = 0;
    // 嘴部张开（打哈欠）持续时长
    bool     mouth_open = false;
    uint64_t mouth_open_start_ms = 0;
    // 头部低垂（点头）持续时长
    bool     head_dropping = false;
    uint64_t head_drop_start_ms = 0;

    // 当前帧最新值
    float cur_ear = 0.3f;
    float cur_ear_smoothed = 0.3f;  // EMA 平滑后的 EAR
    float cur_mar = 0;
    float cur_pitch = 0;
    uint64_t cur_pts = 0;

    // 自校准动态阈值（运行期持续更新，不锁死）
    float cur_ear_thresh = 0.20f;   // 运行时闭眼阈值 = baseline - offset
    bool  calibrated     = false;   // 初始校准是否完成
};

// 判定策略接口（策略模式）
// 每个策略根据上下文 + 当前帧特征，决定 AlarmLevel + reason
class IFatigueStrategy {
public:
    virtual ~IFatigueStrategy() = default;
    virtual const char* name() const = 0;
    // 返回 AlarmLevel::kNone 表示本策略未触发
    virtual AlarmLevel evaluate(const FatigueContext& ctx,
                                std::string& reason) = 0;
};

// 模板方法模式：定义"特征提取 → 时序累积 → 疲劳计算 → 触发判定"标准流程
// 子类可重写 extract_features / compute_metrics / aggregate_decisions，
// 但整体骨架（顺序、调用时机）保持固定。
class FatigueAnalyzer {
public:
    FatigueAnalyzer();
    ~FatigueAnalyzer();

    // 配置
    struct Config {
        float ear_close_thresh = 0.20f;   // 闭眼判定阈值
        float mar_yawn_thresh  = 0.50f;   // 打哈欠阈值
        float pitch_drop_thresh = -15.0f; // 头部低垂阈值（度）
        uint64_t perclos_window_ms = 60000;
        float perclos_warn_thresh  = 0.15f;  // PERCLOS ≥ 15% → WARN（70% 标准）
        float perclos_danger_thresh = 0.30f;
        uint64_t eye_close_warn_ms   = 1500; // 闭眼≥1.5s 提示
        uint64_t eye_close_danger_ms = 3000; // 闭眼≥3s 危险
        uint64_t yawn_warn_ms        = 2000; // 哈欠≥2s 提示
        uint64_t head_drop_warn_ms   = 2000;
        bool enable_strategy_perclos = true;
        bool enable_strategy_closure = true;
        bool enable_strategy_yawn    = true;
        bool enable_strategy_head    = true;
    };
    void set_config(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    // 添加判定策略（可插拔，多个策略并行评估，取最高级别）
    void add_strategy(std::unique_ptr<IFatigueStrategy> s);

    // 输入一帧检测结果（驱动整个模板流程），输出当前帧的 AlarmEvent
    // event.level == kNone 表示当前帧无预警
    AlarmEvent process(const DetectionResult& det);

    // 诊断
    const FatigueContext& context() const { return ctx_; }

private:
    // ============== 模板方法：固定流程，子类可 override ==============
    // Step1：从检测帧提取特征（EAR/MAR/姿态），更新 ctx_.cur_*
    virtual void extract_features(const DetectionResult& det);
    // Step2：时序累积：滑窗、闭眼起止计时、眨眼计数
    virtual void accumulate_timeline();
    // Step3：计算 PERCLOS / 眨眼频率等聚合指标
    virtual void compute_metrics();
    // Step4：调用各策略综合判定，取最高级别
    virtual AlarmLevel aggregate_decisions(std::string& reason, float& metric);

    Config cfg_;
    FatigueContext ctx_;
    std::vector<std::unique_ptr<IFatigueStrategy>> strategies_;

    // 眨眼计数：从"闭眼→睁眼"为一次眨眼
    uint64_t blink_count_ = 0;
    uint64_t blink_window_start_ms_ = 0;
    
    // 防抖计数：连续帧计数器
    int eye_closed_frame_count_ = 0;
    int eye_open_frame_count_ = 0;

    // ===== 自校准：动态 EAR 阈值（持续适应，不锁死）=====
    float ear_baseline_ = 0.30f;                 // 睁眼 EAR 基线
    std::deque<float> calib_ears_;               // 初始校准期 EAR 采集
    std::deque<float> recent_open_ears_;         // 运行期"睁眼EAR"，用于基线滚动刷新
    size_t frame_since_refresh_ = 0;            // 距上次 baseline 刷新的帧数
    static constexpr size_t kCalibFrames             = 30;    // 初始校准帧数（~2s@15fps）
    static constexpr float  kEarOffset               = 0.06f; // 闭眼阈值 = baseline - offset
    static constexpr size_t kBaselineRefreshInterval = 60;    // 每60帧刷新baseline
};

}  // namespace dms

#endif  // DMS_FATIGUE_ANALYZER_H
