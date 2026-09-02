#ifndef DMS_MOCK_DETECTOR_H
#define DMS_MOCK_DETECTOR_H

#include <cstdint>

#include "ai/detector.h"

namespace dms {

// Mock 检测器：不依赖 RKNN，根据"眨眼周期"合成人脸框 + 关键点 + EAR
//
// 用途：
// 1. 开发机（无 NPU）跑通整条流水线
// 2. 测试 PERCLOS / 疲劳分析算法：周期性眨眼 + 可注入的"疲劳"状态
//
// 行为：
// - 每帧返回画面中心一个固定 bbox
// - EAR 按正弦周期变化（每 3 秒眨眼一次 → 闭眼 ~150ms）
// - 可手动设置 simulated_fatigue=true，让 EAR 长时间偏低模拟疲劳
class MockDetector : public Detector {
public:
    bool init() override { return true; }

    void input_shape(uint32_t& w, uint32_t& h) const override {
        w = model_w_; h = model_h_;
    }

    bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                DetectionResult& result) override;

    const char* name() const override { return "MockDetector"; }

    // 测试用：注入模拟疲劳（true=长期闭眼）
    void set_simulated_fatigue(bool v) { simulated_fatigue_ = v; }

private:
    uint32_t model_w_ = 320;
    uint32_t model_h_ = 240;
    uint64_t frame_idx_ = 0;
    bool simulated_fatigue_ = false;
};

}  // namespace dms

#endif  // DMS_MOCK_DETECTOR_H
