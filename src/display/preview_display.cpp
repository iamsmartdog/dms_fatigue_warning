#include "display/preview_display.h"

#include <chrono>

#ifdef DMS_HAS_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "utils/log.h"

namespace dms {

PreviewDisplay::~PreviewDisplay() { stop(); }

bool PreviewDisplay::start(const std::string& window_title) {
    if (running_.exchange(true)) return true;
    title_ = window_title;
    render_thread_ = std::thread(&PreviewDisplay::render_loop, this);
    LOGI("DISP", "preview display started: %s", title_.c_str());
    return true;
}

void PreviewDisplay::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (render_thread_.joinable()) render_thread_.join();
#ifdef DMS_HAS_OPENCV
    if (gui_ok_) {
        cv::destroyAllWindows();
    }
#endif
    LOGI("DISP", "preview display stopped");
}

void PreviewDisplay::update(const uint8_t* rgb, uint32_t w, uint32_t h,
                            const DetectionResult& det,
                            AlarmLevel level, const std::string& reason,
                            float ear_thresh) {
    if (!running_.load() || !gui_ok_) return;

    // 双缓冲：拷贝到 pending_，渲染线程消费
    std::unique_lock<std::mutex> lock(mtx_, std::try_to_lock);
    if (!lock.owns_lock()) return;  // 渲染线程持有锁，直接丢弃本帧（不阻塞推理）

    if (!pending_) pending_ = std::make_unique<Frame>();
    pending_->rgb.assign(rgb, rgb + static_cast<size_t>(w) * h * 3);
    pending_->w = w; pending_->h = h;
    pending_->det = det;
    pending_->level = level;
    pending_->reason = reason;
    pending_->ear_thresh = ear_thresh;
    has_new_.store(true);
    cv_.notify_one();
}

#ifdef DMS_HAS_OPENCV

void PreviewDisplay::render_loop() {
    LOGI("DISP", "render thread tid=%u", static_cast<unsigned>(::gettid()));

    // 首次创建窗口（在渲染线程，OpenCV 要求 GUI 操作在同一线程）
    try {
        cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
    } catch (const cv::Exception& e) {
        LOGE("DISP", "namedWindow failed: %s (no display?)", e.what());
        gui_ok_ = false;
        return;
    }

    while (running_.load()) {
        std::unique_ptr<Frame> frame;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait_for(lock, std::chrono::milliseconds(33),
                         [this] { return has_new_.load() || !running_.load(); });
            if (!running_.load()) break;
            if (has_new_.load()) {
                frame = std::move(pending_);
                has_new_.store(false);
            }
        }
        if (!frame) continue;  // 超时无新帧

        // RGB888 -> BGR888 (OpenCV 默认)
        cv::Mat img(static_cast<int>(frame->h), static_cast<int>(frame->w),
                    CV_8UC3);
        for (uint32_t i = 0; i < frame->w * frame->h; ++i) {
            img.data[i * 3 + 0] = frame->rgb[i * 3 + 2];  // B
            img.data[i * 3 + 1] = frame->rgb[i * 3 + 1];  // G
            img.data[i * 3 + 2] = frame->rgb[i * 3 + 0];  // R
        }

        // 叠加人脸框 + EAR
        bool eyes_closed = false;
        for (const auto& f : frame->det.faces) {
            cv::Rect r(static_cast<int>(f.x), static_cast<int>(f.y),
                       static_cast<int>(f.w), static_cast<int>(f.h));
            float ear = (f.ear_left + f.ear_right) / 2.0f;
            eyes_closed = (ear < frame->ear_thresh);

            cv::Scalar color = eyes_closed ? cv::Scalar(0, 0, 255)   // 红
                                           : cv::Scalar(0, 255, 0);  // 绿
            cv::rectangle(img, r, color, 2);

            // EAR 标签 + 眼睛检测计数（det=N/15，N越大说明睁眼越稳定）
            char buf[80];
            snprintf(buf, sizeof(buf), "EAR=%.2f %s det=%d/15",
                     ear, eyes_closed ? "[CLOSED]" : "[OPEN]",
                     static_cast<int>(f.score));
            cv::putText(img, buf, cv::Point(r.x, r.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // 顶部状态栏
        char status[128];
        const char* lv_str = alarm_level_str(frame->level);
        snprintf(status, sizeof(status), "Alarm: %s | %s",
                 lv_str, frame->reason.c_str());
        cv::Scalar status_color = cv::Scalar(255, 255, 255);
        if (frame->level == AlarmLevel::kWarn)   status_color = cv::Scalar(0, 165, 255);
        if (frame->level == AlarmLevel::kDanger) status_color = cv::Scalar(0, 0, 255);
        cv::rectangle(img, cv::Rect(0, 0, img.cols, 28),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(img, status, cv::Point(5, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1);

        // 闭眼/危险时全屏红色边框（已禁用，不影响人脸框）
        (void)eyes_closed;
        (void)frame;

        cv::imshow(title_, img);
        // waitKey 必须周期性调用，否则窗口不刷新
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {  // ESC / q 退出
            LOGI("DISP", "exit key pressed, stopping");
            running_.store(false);
            break;
        }
    }
}

#else  // !DMS_HAS_OPENCV

void PreviewDisplay::render_loop() {
    static bool warned = false;
    if (!warned) {
        LOGW("DISP", "[STUB] OpenCV not linked, preview disabled");
        warned = true;
    }
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

#endif  // DMS_HAS_OPENCV

}  // namespace dms
