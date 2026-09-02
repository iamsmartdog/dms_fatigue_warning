#ifndef DMS_PIPELINE_STAGE_H
#define DMS_PIPELINE_STAGE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "utils/log.h"
#include "utils/media_types.h"
#include "utils/ring_buffer.h"

namespace dms {

// ============================================================
// PipelineStage：五级线程流水线的统一抽象
// ------------------------------------------------------------
// 设计要点：
// 1. 每个 stage 一个独立线程，从上游 RingBuffer pop、处理、push 到下游
// 2. 模板方法模式：on_process() 由子类实现具体逻辑
// 3. 支持优雅停止：stop() 设置 running_=false 并唤醒所有阻塞的 pop
// 4. 统一采集统计（处理帧数 / 丢弃数 / 耗时），便于性能分析
// ============================================================

class PipelineStageBase {
public:
    explicit PipelineStageBase(std::string name)
        : name_(std::move(name)) {}
    virtual ~PipelineStageBase() { stop(); }

    // 子类实现：启动前的初始化（打开设备/加载模型/创建编码器）
    virtual bool init() = 0;

    // 启动工作线程
    virtual bool start() {
        if (running_.exchange(true)) {
            LOGW(tag(), "already running");
            return true;
        }
        if (!init()) {
            running_ = false;
            LOGE(tag(), "init failed, cannot start");
            return false;
        }
        thread_ = std::thread(&PipelineStageBase::run, this);
        LOGI(tag(), "stage started");
        return true;
    }

    // 优雅停止
    virtual void stop() {
        if (!running_.exchange(false)) return;
        on_stop_requested();
        if (thread_.joinable()) thread_.join();
        LOGI(tag(), "stage stopped, processed=%lu", processed_.load());
    }

    const std::string& name() const { return name_; }
    const char*        tag()  const { return name_.c_str(); }
    uint64_t processed_count() const { return processed_.load(); }

protected:
    // 子类实现：单次数据处理逻辑（阻塞点）
    virtual void process() = 0;

    // 子类实现：被要求停止时，唤醒可能在阻塞的 pop（如 ring_->stop()）
    virtual void on_stop_requested() {}

    std::string         name_;
    std::atomic<bool>   running_{false};
    std::thread         thread_;
    std::atomic<uint64_t> processed_{0};

private:
    void run() {
        LOGI(tag(), "worker thread tid=%u started",
             static_cast<unsigned>(::gettid()));
        while (running_.load()) {
            process();
        }
        LOGI(tag(), "worker thread exited");
    }
};

// ============================================================
// 通用的"输入 -> 输出"线程级，单生产单消费场景
// InT  : 输入元素类型（VideoFrame / AudioFrame / DetectionResult ...）
// 嵌入一个 RingBuffer 作为输入队列
// ============================================================
template <typename InT>
class SourceStage : public PipelineStageBase {
public:
    SourceStage(const std::string& name, size_t ring_cap)
        : PipelineStageBase(name)
        , in_(std::make_unique<RingBuffer<InT>>(ring_cap)) {}

    // 上游 push 入口
    int push_input(const InT& item) { return in_->push(item); }
    RingBuffer<InT>& input_queue() { return *in_; }

protected:
    // 阻塞取一帧（停止/超时返回 false）
    bool take(InT& out, int timeout_ms = -1) {
        return in_->pop(out, timeout_ms);
    }

    void on_stop_requested() override { in_->stop(); }

    std::unique_ptr<RingBuffer<InT>> in_;
};

}  // namespace dms

#endif  // DMS_PIPELINE_STAGE_H
