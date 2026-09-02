#ifndef DMS_PIPELINE_H
#define DMS_PIPELINE_H

#include <atomic>
#include <memory>
#include <thread>

#include "ai/detector.h"
#include "alarm/alarm_manager.h"
#include "alarm/event_recorder.h"
#include "alarm/fatigue_analyzer.h"
#include "audio/alsa_capture.h"
#include "capture/v4l2_capture.h"
#include "display/preview_display.h"
#include "encoder/aac_encoder.h"
#include "encoder/rkmpp_encoder.h"
#include "muxer/mp4_muxer.h"
#include "preprocess/image_preprocess.h"
#include "utils/media_types.h"

namespace dms {

// 完整 DMS 流水线编排
//
// 8 个并发线程（对应设计文档线程模型）：
//   T1 capture_video : V4L2 DQBUF → video_ring
//   T2 capture_audio : ALSA      → audio_ring
//   T3 preprocess    : video_ring → 预处理(NV12/RGB) → 双输出
//                       ├→ enc_video_ring (喂编码器)
//                       └→ ai_ring (每隔 N 帧喂推理，降帧)
//   T4 inference     : ai_ring → DetectionResult → 疲劳分析 → AlarmEvent
//   T5 encode_video  : enc_video_ring → RKMPP → EncodedPacket → recorder
//   T6 encode_audio  : audio_ring → AAC → EncodedPacket → recorder
//   T7 alarm         : 处理 AlarmEvent（冷却/TTS/触发录像）
//   主线程：监控统计、信号处理
//
// 组件持有关系：所有模块以 unique_ptr 持有，Pipeline 析构时按倒序停止
class DmsPipeline {
public:
    DmsPipeline();
    ~DmsPipeline();

    struct Config {
        // 视频
        std::string video_device  = "/dev/video0";
        uint32_t    video_width   = 1280;
        uint32_t    video_height  = 720;
        uint32_t    video_fps     = 30;
        PixelFormat video_format  = PixelFormat::kMjpg;
        // 音频
        std::string audio_device  = "default";
        uint32_t    audio_rate    = 16000;
        uint32_t    audio_channels = 1;
        // 编码
        uint32_t    enc_bitrate   = 2000000;
        uint32_t    enc_gop       = 60;
        // AI
        // detector_type: "mock"=MockDetector, "cv"=OpenCV Haar, "yunet"=YuNet, "rknn"=RknnDetector
        std::string detector_type = "mock";
        std::string model_path;            // rknn 模型路径 / cv 级联文件目录 / yunet 模型路径
        std::string pfld_model_path;       // PFLD 106 点关键点模型（可选，仅 yunet 用）
        uint32_t    model_w = 320;
        uint32_t    model_h = 240;
        uint32_t    inference_fps = 15;    // NPU 推理降帧：30fps 采集中只推 15fps
        // 输出
        std::string record_dir = "./records";
        bool        encrypt = true;
        bool        stub_dump_first_frames = false;  // 调试：保存前 N 帧原图
        // 调试显示
        bool        enable_preview = false;  // 虚拟机/开发机 imshow 弹窗
        std::string preview_title  = "DMS Preview";
    };

    bool init(const Config& cfg);
    bool start();
    void stop();
    void wait();   // 阻塞直到外部 signal 停止

    // 统计快照（主线程定期打印）
    struct Stats {
        uint64_t v_captured = 0, v_dropped = 0;
        uint64_t a_captured = 0, a_dropped = 0;
        uint64_t v_encoded  = 0;
        uint64_t a_encoded  = 0;
        uint64_t inferred   = 0;
        uint64_t info_cnt = 0, warn_cnt = 0, danger_cnt = 0;
        float cur_perclos = 0;
        float cur_ear = 0;
    };
    Stats stats() const;

private:
    // 各工作线程入口
    void video_capture_thread();
    void audio_capture_thread();
    void preprocess_thread();
    void inference_thread();
    void encode_video_thread();
    void encode_audio_thread();

    Config cfg_;
    std::atomic<bool> running_{false};

    // 模块
    std::unique_ptr<V4l2Capture>     v4l2_;
    std::unique_ptr<AlsaCapture>     alsa_;
    std::unique_ptr<ImagePreprocess> preproc_;
    std::unique_ptr<Detector>        detector_;
    std::unique_ptr<RkmppEncoder>    venc_;
    std::unique_ptr<AacEncoder>      aenc_;
    std::unique_ptr<FatigueAnalyzer> analyzer_;
    std::unique_ptr<AlarmManager>    alarm_;
    std::unique_ptr<EventRecorder>   recorder_;
    std::unique_ptr<PreviewDisplay>  display_;

    // 线程间环形缓冲（VideoFrame 不可拷贝，用 shared_ptr 包装）
    std::unique_ptr<RingBuffer<std::shared_ptr<VideoFrame>>> raw_video_ring_;
    std::unique_ptr<RingBuffer<AudioFrame>>              raw_audio_ring_;
    std::unique_ptr<RingBuffer<std::shared_ptr<VideoFrame>>> enc_video_ring_;
    // 推理分支：直接用 RGB 输入缓冲
    struct InferenceInput {
        std::vector<uint8_t> rgb;
        uint32_t w = 0, h = 0;
        uint64_t pts_ms = 0;
    };
    std::unique_ptr<RingBuffer<InferenceInput>> ai_ring_;

    std::thread t_vcap_, t_acap_, t_prep_, t_inf_, t_venc_, t_aenc_;

    // NPU 降帧计数
    std::atomic<uint32_t> ai_skip_cnt_{0};

    // 实时共享指标（供 stats）
    std::atomic<uint64_t> v_encoded_{0}, a_encoded_{0}, inferred_{0};
    std::atomic<float> cur_perclos_{0}, cur_ear_{0};
    // 当前报警级别 + 原因（供 display 读取，原子读写无锁）
    std::atomic<int>      cur_alarm_level_{0};  // AlarmLevel 转 int
    std::mutex            alarm_reason_mtx_;
    std::string           cur_alarm_reason_;
};

}  // namespace dms

#endif  // DMS_PIPELINE_H
