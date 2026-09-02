#ifndef DMS_V4L2_CAPTURE_H
#define DMS_V4L2_CAPTURE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/ring_buffer.h"
#include "utils/media_types.h"   // PixelFormat（统一定义在 media_types.h）

// 前向声明，避免头文件依赖 linux/videodev2.h
struct v4l2_buffer;

namespace dms {

// 采集参数
struct CaptureConfig {
    std::string  device      = "/dev/video0";
    uint32_t     width       = 1280;       // 720P: 1280x720, 1080P: 1920x1080
    uint32_t     height      = 720;
    uint32_t     fps         = 30;
    PixelFormat  format      = PixelFormat::kMjpg;
    uint32_t     buffer_count = 4;         // V4L2 mmap 缓冲区数量
    size_t       ring_capacity = 8;        // 环形帧缓冲容量
};

// 采集到的帧（拥有数据所有权，深拷贝自 mmap，安全跨线程传递）
// 修复 use-after-free：原实现 Frame.data 指向 mmap 内存，QBUF 归还后
// 内核可立即覆写该区域，导致消费端拷贝到被覆写的数据
struct CapturedFrame {
    std::vector<uint8_t> data;   // 帧数据（深拷贝，独立于 mmap 缓冲区）
    uint64_t pts_ms = 0;
    uint32_t seq    = 0;
    bool is_valid   = false;
};

// V4L2 单设备采集器
//
// 底层实现要点：
// 1. 直接 ioctl 配置设备（不依赖 OpenCV 等高层接口）
// 2. mmap 方式管理内核缓冲区，零拷贝获取帧数据
// 3. 内部采集线程负责 DQBUF→入环形缓冲→QBUF
// 4. 帧带 PTS 时间戳，为后续音视频同步打基础
class V4l2Capture {
public:
    V4l2Capture();
    ~V4l2Capture();

    V4l2Capture(const V4l2Capture&) = delete;
    V4l2Capture& operator=(const V4l2Capture&) = delete;

    // 打开设备并查询能力
    bool open(const std::string& device);

    // 配置采集参数（格式/分辨率/帧率/缓冲区）
    bool configure(const CaptureConfig& cfg);

    // 启动采集线程
    bool start();

    // 停止采集
    void stop();

    // 从环形缓冲获取一帧（阻塞，timeout_ms=-1 永久等待）
    bool get_frame(CapturedFrame& frame, int timeout_ms = -1);

    // 查询设备支持的格式（调试用）
    void list_formats();

    // 统计信息
    uint64_t captured_count() const { return captured_.load(); }
    uint64_t dropped_count()   const { return dropped_.load(); }

private:
    // 内部采集线程主循环
    void capture_loop();

    // V4L2 ioctl 封装（自动重试 EINTR）
    // request 类型用 unsigned long 匹配 ioctl(2) 原型，避免 VIDIOC_* 宏溢出警告
    int xioctl(int fd, unsigned long request, void* arg);

    // 设置像素格式与分辨率
    bool set_format();
    // 设置帧率
    bool set_framerate();
    // 申请并 mmap 内核缓冲区
    bool init_mmap_buffers();
    // 入队所有缓冲区
    bool queue_all_buffers();

    // mmap 缓冲区描述
    struct MmapBuffer {
        void*   start  = nullptr;
        size_t  length = 0;
    };

    int                   fd_        = -1;
    CaptureConfig         cfg_;
    std::vector<MmapBuffer> mmap_buffers_;
    std::unique_ptr<RingBuffer<CapturedFrame>> ring_;
    std::thread           capture_thread_;
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> captured_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint32_t> seq_{0};

    static constexpr const char* kTag = "V4L2";
};

}  // namespace dms

#endif  // DMS_V4L2_CAPTURE_H
