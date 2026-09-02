#ifndef DMS_RING_BUFFER_H
#define DMS_RING_BUFFER_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

namespace dms {

// 带时间戳的帧数据
struct Frame {
    void*   data      = nullptr;  // 帧数据指针（可能指向 mmap 内存，不持有所有权）
    size_t  size      = 0;        // 帧字节数
    uint64_t pts_ms   = 0;        // 时间戳（毫秒，用于音视频同步）
    uint32_t seq      = 0;        // 序列号（丢帧检测）
    bool    is_valid  = false;    // 是否有效帧

    void copy_to(void* dst, size_t max_len) const {
        if (!data || !dst || size == 0) return;
        size_t n = (size < max_len) ? size : max_len;
        memcpy(dst, data, n);
    }
};

// 线程安全环形帧缓冲区（生产者-消费者模型）
//
// 设计要点：
// 1. 实时采集场景下，丢帧优于阻塞——缓冲满时丢弃最老帧，保证采集线程不阻塞
// 2. 消费端可阻塞等待新帧，也可 try_pop 非阻塞获取
// 3. 通过 PTS + 序列号支持丢帧统计与音视频同步
template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity), head_(0), tail_(0), count_(0), stopped_(false) {
        buffer_.resize(capacity);
    }

    ~RingBuffer() { stop(); }

    // 生产者：写入一帧。缓冲满时丢弃最老帧（实时策略）
    // 返回丢弃的帧数（0 表示未丢帧）
    int push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        int dropped = 0;
        if (count_ == capacity_) {
            // 丢最老帧：head 前移
            head_ = (head_ + 1) % capacity_;
            --count_;
            ++dropped;
        }
        buffer_[tail_] = item;
        tail_ = (tail_ + 1) % capacity_;
        ++count_;
        cv_.notify_one();
        return dropped;
    }

    // 消费者：阻塞获取一帧，返回 false 表示缓冲区已停止
    bool pop(T& out, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (timeout_ms < 0) {
            cv_.wait(lock, [this] { return count_ > 0 || stopped_; });
        } else {
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return count_ > 0 || stopped_; });
        }
        if (count_ == 0) return false;  // stopped 或超时

        out = buffer_[head_];
        head_ = (head_ + 1) % capacity_;
        --count_;
        return true;
    }

    // 非阻塞获取
    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (count_ == 0) return false;
        out = buffer_[head_];
        head_ = (head_ + 1) % capacity_;
        --count_;
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
        cv_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx_);
        head_ = tail_ = count_ = 0;
        stopped_ = false;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

    size_t capacity() const { return capacity_; }

private:
    std::vector<T>   buffer_;
    const size_t     capacity_;
    size_t           head_;     // 消费位置
    size_t           tail_;     // 生产位置
    size_t           count_;    // 当前帧数
    bool             stopped_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};

}  // namespace dms

#endif  // DMS_RING_BUFFER_H
