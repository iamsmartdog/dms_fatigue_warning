#include "capture/v4l2_capture.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

#include "utils/log.h"

namespace dms {

V4l2Capture::V4l2Capture() = default;

V4l2Capture::~V4l2Capture() {
    stop();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    // 释放 mmap 内存
    for (auto& buf : mmap_buffers_) {
        if (buf.start != nullptr && buf.start != MAP_FAILED) {
            ::munmap(buf.start, buf.length);
        }
    }
}

// ioctl 封装：EINTR 自动重试
int V4l2Capture::xioctl(int fd, unsigned long request, void* arg) {
    int ret = 0;
    do {
        ret = ::ioctl(fd, request, arg);
    } while (ret == -1 && errno == EINTR);
    return ret;
}

bool V4l2Capture::open(const std::string& device) {
    // O_RDWR 读写 | O_NONBLOCK 非阻塞（配合 select/poll）
    // 注意：此处用阻塞模式，采集线程内 DQBUF 阻塞等待
    fd_ = ::open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        LOGE(kTag, "open %s failed: %s", device.c_str(), strerror(errno));
        return false;
    }

    // 查询设备能力
    struct v4l2_capability cap = {};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        LOGE(kTag, "VIDIOC_QUERYCAP failed: %s", strerror(errno));
        return false;
    }

    // 检查是否为视频采集设备
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        LOGE(kTag, "%s is not a video capture device", device.c_str());
        return false;
    }

    // 检查是否支持流式IO（mmap）
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        LOGE(kTag, "%s does not support streaming i/o", device.c_str());
        return false;
    }

    LOGI(kTag, "device: %s", cap.card);
    LOGI(kTag, "driver: %s", cap.driver);
    LOGI(kTag, "bus:    %s", cap.bus_info);
    return true;
}

bool V4l2Capture::set_format() {
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = cfg_.width;
    fmt.fmt.pix.height      = cfg_.height;
    fmt.fmt.pix.pixelformat = static_cast<uint32_t>(cfg_.format);
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;  // 逐行扫描

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        LOGE(kTag, "VIDIOC_S_FMT failed: %s", strerror(errno));
        return false;
    }

    // 驱动可能调整了请求的参数，检查实际值
    if (fmt.fmt.pix.width != cfg_.width || fmt.fmt.pix.height != cfg_.height) {
        LOGW(kTag, "requested %ux%u, got %ux%u (driver adjusted)",
             cfg_.width, cfg_.height, fmt.fmt.pix.width, fmt.fmt.pix.height);
        cfg_.width  = fmt.fmt.pix.width;
        cfg_.height = fmt.fmt.pix.height;
    }
    if (fmt.fmt.pix.pixelformat != static_cast<uint32_t>(cfg_.format)) {
        char got[5] = {};
        memcpy(got, &fmt.fmt.pix.pixelformat, 4);
        LOGW(kTag, "requested format not supported, driver gave: %s", got);
        cfg_.format = static_cast<PixelFormat>(fmt.fmt.pix.pixelformat);
    }

    LOGI(kTag, "format: %ux%u, sizeimage=%u",
         cfg_.width, cfg_.height, fmt.fmt.pix.sizeimage);
    return true;
}

bool V4l2Capture::set_framerate() {
    struct v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = cfg_.fps;

    if (xioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        LOGW(kTag, "VIDIOC_S_PARM failed: %s (non-fatal)", strerror(errno));
        return true;  // 部分设备不支持设帧率，非致命
    }

    uint32_t actual_fps = parm.parm.capture.timeperframe.denominator /
                          parm.parm.capture.timeperframe.numerator;
    if (actual_fps != cfg_.fps) {
        LOGW(kTag, "requested %u fps, got %u fps", cfg_.fps, actual_fps);
        cfg_.fps = actual_fps;
    }
    return true;
}

bool V4l2Capture::init_mmap_buffers() {
    // 申请内核缓冲区（MMAP 模式）
    struct v4l2_requestbuffers req = {};
    req.count  = cfg_.buffer_count;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        LOGE(kTag, "VIDIOC_REQBUFS failed: %s", strerror(errno));
        return false;
    }

    if (req.count < 2) {
        LOGE(kTag, "Insufficient buffer memory: %u", req.count);
        return false;
    }
    cfg_.buffer_count = req.count;
    LOGI(kTag, "allocated %u kernel buffers", req.count);

    mmap_buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf = {};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            LOGE(kTag, "VIDIOC_QUERYBUF[%u] failed: %s", i, strerror(errno));
            return false;
        }

        mmap_buffers_[i].length = buf.length;
        mmap_buffers_[i].start  = ::mmap(nullptr, buf.length,
                                         PROT_READ | PROT_WRITE,
                                         MAP_SHARED, fd_, buf.m.offset);
        if (mmap_buffers_[i].start == MAP_FAILED) {
            LOGE(kTag, "mmap[%u] failed: %s", i, strerror(errno));
            return false;
        }
    }
    return true;
}

bool V4l2Capture::queue_all_buffers() {
    for (uint32_t i = 0; i < cfg_.buffer_count; ++i) {
        struct v4l2_buffer buf = {};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            LOGE(kTag, "VIDIOC_QBUF[%u] failed: %s", i, strerror(errno));
            return false;
        }
    }
    return true;
}

bool V4l2Capture::configure(const CaptureConfig& cfg) {
    cfg_ = cfg;

    if (!set_format())      return false;
    if (!set_framerate())   return false;
    if (!init_mmap_buffers()) return false;
    if (!queue_all_buffers()) return false;

    // 创建环形帧缓冲
    ring_ = std::make_unique<RingBuffer<CapturedFrame>>(cfg_.ring_capacity);
    LOGI(kTag, "configured: %ux%u@%ufps, ring=%zu",
         cfg_.width, cfg_.height, cfg_.fps, cfg_.ring_capacity);
    return true;
}

bool V4l2Capture::start() {
    if (running_.load()) {
        LOGW(kTag, "already running");
        return true;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        LOGE(kTag, "VIDIOC_STREAMON failed: %s", strerror(errno));
        return false;
    }

    running_ = true;
    ring_->reset();
    captured_ = 0;
    dropped_  = 0;
    seq_      = 0;
    capture_thread_ = std::thread(&V4l2Capture::capture_loop, this);
    LOGI(kTag, "streaming started");
    return true;
}

void V4l2Capture::stop() {
    if (!running_.exchange(false)) return;

    // 先 STREAMOFF 使阻塞的 DQBUF 返回错误，再 join 线程
    // （若先 join，采集线程阻塞在 DQBUF 上会导致永久挂死）
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (fd_ >= 0) {
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    if (ring_) ring_->stop();

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    LOGI(kTag, "streaming stopped, captured=%lu dropped=%lu",
         captured_.load(), dropped_.load());
}

void V4l2Capture::capture_loop() {
    LOGI(kTag, "capture thread started, tid=%u",
         static_cast<unsigned>(::gettid()));

    while (running_.load()) {
        struct v4l2_buffer buf = {};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // DQBUF 阻塞等待一帧就绪
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) continue;       // 非阻塞模式无数据
            if (errno == EIO)   continue;        // 可忽略的IO错误
            LOGE(kTag, "VIDIOC_DQBUF failed: %s", strerror(errno));
            break;
        }

        // 检查帧标志
        if (buf.flags & V4L2_BUF_FLAG_ERROR) {
            LOGW(kTag, "frame[%u] has error, skipped", buf.index);
        }

        // 深拷贝帧数据（必须在 QBUF 前完成：QBUF 归还后内核可立即覆写 mmap 内存）
        CapturedFrame frame;
        size_t copy_len = buf.bytesused;
        if (copy_len > mmap_buffers_[buf.index].length) {
            copy_len = mmap_buffers_[buf.index].length;  // 边界保护
        }
        frame.data.resize(copy_len);
        std::memcpy(frame.data.data(), mmap_buffers_[buf.index].start, copy_len);
        // PTS：优先用 V4L2 时间戳，回退到系统时间
        if (buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) {
            frame.pts_ms = static_cast<uint64_t>(buf.timestamp.tv_sec) * 1000 +
                           buf.timestamp.tv_usec / 1000;
        } else {
            frame.pts_ms = now_ms();
        }
        frame.seq      = seq_.fetch_add(1);
        frame.is_valid = !(buf.flags & V4L2_BUF_FLAG_ERROR);

        // 将缓冲区重新入队（帧数据已深拷贝，内核可安全覆写 mmap）
        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            LOGE(kTag, "VIDIOC_QBUF failed: %s", strerror(errno));
            break;
        }

        // 入环形缓冲（满则丢老帧）
        int dropped = ring_->push(std::move(frame));
        if (dropped > 0) dropped_.fetch_add(dropped);
        captured_.fetch_add(1);
    }

    LOGI(kTag, "capture thread exited");
}

bool V4l2Capture::get_frame(CapturedFrame& frame, int timeout_ms) {
    if (!ring_) return false;
    return ring_->pop(frame, timeout_ms);
}

void V4l2Capture::list_formats() {
    struct v4l2_fmtdesc fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.index = 0;

    LOGI(kTag, "supported formats:");
    while (xioctl(fd_, VIDIOC_ENUM_FMT, &fmt) == 0) {
        char fourcc[5] = {};
        memcpy(fourcc, &fmt.pixelformat, 4);
        LOGI(kTag, "  [%s] %s", fourcc, fmt.description);

        // 枚举该格式支持的分辨率
        struct v4l2_frmsizeenum frmsize = {};
        frmsize.pixel_format = fmt.pixelformat;
        frmsize.index = 0;
        while (xioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                LOGI(kTag, "    %ux%u",
                     frmsize.discrete.width, frmsize.discrete.height);
            }
            ++frmsize.index;
        }
        ++fmt.index;
    }
}

}  // namespace dms
