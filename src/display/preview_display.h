#ifndef DMS_PREVIEW_DISPLAY_H
#define DMS_PREVIEW_DISPLAY_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "utils/media_types.h"

namespace dms {

// 实时画面预览（开发机/虚拟机调试用）
//
// 职责：
// 1. 接收推理分支的 RGB 帧 + 检测结果，用 OpenCV imshow 弹窗显示
// 2. 叠加：人脸框、EAR 值、闭眼/睁眼状态、报警级别
// 3. 闭眼时画面变红边框 + "EYES CLOSED!" 提示，WARN/DANGER 时全屏红色闪烁
//
// 线程模型：
// - update() 由推理线程调用（异步），内部仅做一次浅拷贝 + 唤醒
// - 渲染线程独占运行 imshow（OpenCV GUI 必须在主线程或专用线程）
// - 用 spinlock + atomic_flag 保证更新线程安全
//
// 注意：imshow 在无显示环境（如 SSH 无 X11）会失败，update() 会自动
// 降级为 no-op 并打印一次警告，不影响主流水线
class PreviewDisplay {
public:
    PreviewDisplay() = default;
    ~PreviewDisplay();

    // 启动渲染线程（window_title 为 imshow 窗口标题）
    bool start(const std::string& window_title = "DMS Preview");
    void stop();

    // 推理线程调用：推送一帧画面 + 检测结果 + 当前报警级别
    //   rgb: RGB888 数据（会被内部拷贝）
    //   w/h: 画面尺寸
    //   det: 检测结果（人脸框/EAR）
    //   level: 当前报警级别
    //   reason: 报警原因（可空）
    void update(const uint8_t* rgb, uint32_t w, uint32_t h,
                const DetectionResult& det,
                AlarmLevel level, const std::string& reason,
                float ear_thresh);

private:
    // 渲染线程入口
    void render_loop();

    struct Frame {
        std::vector<uint8_t> rgb;   // RGB888
        uint32_t w = 0, h = 0;
        DetectionResult det;
        AlarmLevel level = AlarmLevel::kNone;
        std::string reason;
        float ear_thresh = 0.20f;   // 当前动态闭眼阈值（与 analyzer 同步）
    };

    std::unique_ptr<Frame>  pending_;      // 待渲染帧（双缓冲，最新覆盖旧）
    std::mutex              mtx_;          // 保护 pending_
    std::condition_variable cv_;
    std::atomic<bool>       running_{false};
    std::atomic<bool>       has_new_{false};
    std::thread             render_thread_;
    bool                    gui_ok_ = true;  // imshow 是否可用（首次失败后置 false）

    std::string             title_ = "DMS Preview";
};

}  // namespace dms

#endif  // DMS_PREVIEW_DISPLAY_H
