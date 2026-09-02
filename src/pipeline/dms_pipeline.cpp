#include "pipeline/dms_pipeline.h"

#include <chrono>
#include <cstring>
#include <fstream>

#include "ai/cv_detector.h"
#include "ai/yunet_detector.h"
#include "ai/mock_detector.h"
#include "ai/rknn_detector.h"
#include "utils/log.h"

namespace dms {

DmsPipeline::DmsPipeline() = default;
DmsPipeline::~DmsPipeline() { stop(); }

bool DmsPipeline::init(const Config& cfg) {
    cfg_ = cfg;

    // ----- 视频采集 -----
    v4l2_ = std::make_unique<V4l2Capture>();
    if (!v4l2_->open(cfg_.video_device)) {
        LOGE("PIPE", "open video device failed");
        return false;
    }
    CaptureConfig vcfg;
    vcfg.device = cfg_.video_device;
    vcfg.width  = cfg_.video_width;
    vcfg.height = cfg_.video_height;
    vcfg.fps    = cfg_.video_fps;
    vcfg.format = cfg_.video_format;
    vcfg.buffer_count = 4;
    vcfg.ring_capacity = 8;
    if (!v4l2_->configure(vcfg)) {
        LOGE("PIPE", "video configure failed");
        return false;
    }

    // ----- 音频采集 -----
    alsa_ = std::make_unique<AlsaCapture>();
    AudioConfig acfg;
    acfg.device = cfg_.audio_device;
    acfg.sample_rate = cfg_.audio_rate;
    acfg.channels = cfg_.audio_channels;
    acfg.period_frames = 1024;
    if (!alsa_->open(acfg)) {
        LOGE("PIPE", "open audio device failed (non-fatal, continuing)");
        // 音频非致命：仍可只跑视频+AI
    }

    // ----- 预处理 -----
    preproc_ = std::make_unique<ImagePreprocess>();
    ImagePreprocess::Config pcfg;
    pcfg.enc_width = cfg_.video_width;
    pcfg.enc_height = cfg_.video_height;
    pcfg.model_width = cfg_.model_w;
    pcfg.model_height = cfg_.model_h;
    pcfg.enable_denoise = true;
    pcfg.enable_contrast_enhance = true;
    preproc_->set_config(pcfg);

    // ----- AI 检测器：按 detector_type 选择 -----
    //   mock : 合成数据，跑通流水线（无摄像头/AI 也可验证）
    //   cv   : OpenCV Haar 人脸 + 眼睛检测（虚拟机/开发机真实检测）
    //   yunet: YuNet 深度学习人脸检测（OpenCV DNN，低光更鲁棒）
    //   rknn : RKNN NPU 推理（板子上用）
    if (cfg_.detector_type == "yunet") {
        detector_ = std::make_unique<YuNetDetector>(cfg_.model_path,
                                                     cfg_.pfld_model_path);
    } else if (cfg_.detector_type == "cv") {
        detector_ = std::make_unique<CvDetector>(cfg_.model_path);
    } else if (cfg_.detector_type == "rknn" || !cfg_.model_path.empty()) {
        detector_ = std::make_unique<RknnDetector>(cfg_.model_path);
    } else {
        detector_ = std::make_unique<MockDetector>();
    }
    if (!detector_->init()) {
        LOGE("PIPE", "detector (%s) init failed", cfg_.detector_type.c_str());
        return false;
    }
    // 同步模型输入尺寸
    detector_->input_shape(pcfg.model_width, pcfg.model_height);
    preproc_->set_config(pcfg);

    // ----- 视频编码器 -----
    venc_ = std::make_unique<RkmppEncoder>();
    EncoderConfig ecfg;
    ecfg.width = cfg_.video_width;
    ecfg.height = cfg_.video_height;
    ecfg.fps = cfg_.video_fps;
    ecfg.gop = cfg_.enc_gop;
    ecfg.bitrate = cfg_.enc_bitrate;
    ecfg.input_fmt = PixelFormat::kNv12;
    ecfg.ring_capacity = 8;
    if (!venc_->open(ecfg)) {
        LOGE("PIPE", "video encoder open failed");
        return false;
    }

    // ----- 音频编码器 -----
    aenc_ = std::make_unique<AacEncoder>();
    if (alsa_) aenc_->open(acfg);

    // ----- 疲劳分析 + 预警 + 录像 -----
    analyzer_ = std::make_unique<FatigueAnalyzer>();
    alarm_ = std::make_unique<AlarmManager>();
    recorder_ = std::make_unique<EventRecorder>();
    EventRecorder::Config rcfg;
    rcfg.output_dir = cfg_.record_dir;
    rcfg.enc_width = cfg_.video_width;
    rcfg.enc_height = cfg_.video_height;
    rcfg.enc_fps = cfg_.video_fps;
    rcfg.audio_sample_rate = cfg_.audio_rate;
    rcfg.audio_channels = cfg_.audio_channels;
    rcfg.encrypt = cfg_.encrypt;
    recorder_->set_config(rcfg);
    recorder_->init();

    // 录像接收：编码器出来的码流直接喂 recorder
    venc_->set_callback([this](const EncodedPacket& pkt) {
        v_encoded_.fetch_add(1);
        recorder_->feed_packet(pkt);
    });
    aenc_->set_callback([this](const EncodedPacket& pkt) {
        a_encoded_.fetch_add(1);
        recorder_->feed_packet(pkt);
    });
    // 预警 → 触发录像
    alarm_->set_callback([this](const AlarmEvent& ev) {
        recorder_->on_alarm(ev);
    });

    // ----- 线程间环形缓冲 -----
    raw_video_ring_ = std::make_unique<RingBuffer<std::shared_ptr<VideoFrame>>>(8);
    raw_audio_ring_ = std::make_unique<RingBuffer<AudioFrame>>(16);
    enc_video_ring_ = std::make_unique<RingBuffer<std::shared_ptr<VideoFrame>>>(8);
    ai_ring_ = std::make_unique<RingBuffer<InferenceInput>>(4);

    // ----- 预览显示（虚拟机/开发机调试用）-----
    if (cfg_.enable_preview) {
        display_ = std::make_unique<PreviewDisplay>();
    }

    LOGI("PIPE", "pipeline initialized: %ux%u@%u, model=%ux%u, recorder=%s, "
         "detector=%s preview=%d",
         cfg_.video_width, cfg_.video_height, cfg_.video_fps,
         cfg_.model_w, cfg_.model_h, cfg_.record_dir.c_str(),
         cfg_.detector_type.c_str(), cfg_.enable_preview ? 1 : 0);
    return true;
}

bool DmsPipeline::start() {
    if (running_.exchange(true)) return true;

    // 重置环形缓冲（清除上次 stop 设置的 stopped_ 状态，否则线程空转）
    if (raw_video_ring_) raw_video_ring_->reset();
    if (raw_audio_ring_) raw_audio_ring_->reset();
    if (ai_ring_)        ai_ring_->reset();
    if (enc_video_ring_) enc_video_ring_->reset();

    // 组件启动顺序：先下游后上游，避免上游丢帧
    if (alsa_) alsa_->start();
    aenc_->start();
    venc_->start();
    v4l2_->start();

    // 工作线程
    t_vcap_  = std::thread(&DmsPipeline::video_capture_thread, this);
    t_acap_  = std::thread(&DmsPipeline::audio_capture_thread, this);
    t_prep_  = std::thread(&DmsPipeline::preprocess_thread, this);
    t_inf_   = std::thread(&DmsPipeline::inference_thread, this);
    t_venc_  = std::thread(&DmsPipeline::encode_video_thread, this);
    t_aenc_  = std::thread(&DmsPipeline::encode_audio_thread, this);

    // 预览显示线程（最后启动，避免没有数据时空转）
    if (display_) display_->start(cfg_.preview_title);

    LOGI("PIPE", "pipeline started (6 worker threads%s)",
         display_ ? " + preview" : "");
    return true;
}

void DmsPipeline::stop() {
    if (!running_.exchange(false)) return;

    // 显示线程先停（避免访问已释放的缓冲）
    if (display_) display_->stop();

    // 上游先停，再停下游，最后 join
    if (v4l2_)  v4l2_->stop();
    if (alsa_)  alsa_->stop();
    if (raw_video_ring_) raw_video_ring_->stop();
    if (raw_audio_ring_) raw_audio_ring_->stop();
    if (ai_ring_)        ai_ring_->stop();
    if (enc_video_ring_) enc_video_ring_->stop();

    if (t_vcap_.joinable()) t_vcap_.join();
    if (t_acap_.joinable()) t_acap_.join();
    if (t_prep_.joinable()) t_prep_.join();
    if (t_inf_.joinable())  t_inf_.join();

    if (venc_) venc_->stop();
    if (aenc_) aenc_->stop();
    if (t_venc_.joinable()) t_venc_.join();
    if (t_aenc_.joinable()) t_aenc_.join();

    if (recorder_) recorder_->flush();
    LOGI("PIPE", "pipeline stopped");
}

void DmsPipeline::wait() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ============== T1 视频采集 ==============
void DmsPipeline::video_capture_thread() {
    LOGI("VCAP", "tid=%u", static_cast<unsigned>(::gettid()));
    int dumped = 0;
    while (running_.load()) {
        CapturedFrame raw;
        if (!v4l2_->get_frame(raw, 200)) continue;

        // CapturedFrame 已在采集线程深拷贝（QBUF 前完成），此处拷入 VideoFrame 供流水线使用
        auto vf = std::make_shared<VideoFrame>();
        vf->copy_from(raw.data.data(), raw.data.size());
        vf->width = cfg_.video_width;
        vf->height = cfg_.video_height;
        vf->format = cfg_.video_format;
        vf->pts_ms = raw.pts_ms;
        vf->seq = raw.seq;
        vf->is_valid = raw.is_valid;

        if (cfg_.stub_dump_first_frames && dumped < 3) {
            char path[128];
            snprintf(path, sizeof(path), "raw_%03d.jpg", dumped++);
            std::ofstream ofs(path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(vf->data), vf->size);
        }
        raw_video_ring_->push(vf);
    }
    LOGI("VCAP", "exited");
}

// ============== T2 音频采集 ==============
void DmsPipeline::audio_capture_thread() {
    LOGI("ACAP", "tid=%u", static_cast<unsigned>(::gettid()));
    if (!alsa_) return;
    while (running_.load()) {
        AudioFrame af;
        if (!alsa_->get_frame(af, 200)) continue;
        raw_audio_ring_->push(af);
    }
    LOGI("ACAP", "exited");
}

// ============== T3 预处理（双输出：编码器 + AI） ==============
void DmsPipeline::preprocess_thread() {
    LOGI("PREP", "tid=%u", static_cast<unsigned>(::gettid()));
    while (running_.load()) {
        std::shared_ptr<VideoFrame> raw;
        if (!raw_video_ring_->pop(raw, 200)) continue;
        if (!raw || !raw->is_valid) continue;

        auto t0 = now_ms();
        auto r = preproc_->process(*raw);
        if (!r.ok) continue;

        // 分支1：喂编码器
        enc_video_ring_->push(r.enc_frame);

        // 分支2：喂 AI（降帧：inference_fps < video_fps 时跳帧）
        // 例：video_fps=30 inference_fps=15 → skip=1，模式为"跳1帧→推1帧"=15fps
        uint32_t skip = (cfg_.video_fps > 0 && cfg_.inference_fps > 0 &&
                         cfg_.inference_fps < cfg_.video_fps)
                        ? cfg_.video_fps / cfg_.inference_fps - 1 : 0;
        if (ai_skip_cnt_.fetch_add(1) < skip) {
            continue;  // 本帧跳过推理
        }
        ai_skip_cnt_.store(0);

        InferenceInput ii;
        ii.rgb = std::move(r.model_input_rgb);
        ii.w = r.model_w; ii.h = r.model_h;
        ii.pts_ms = raw->pts_ms;
        ai_ring_->push(std::move(ii));
    }
    LOGI("PREP", "exited");
}

// ============== T4 推理 + 疲劳分析 + 预警 ==============
void DmsPipeline::inference_thread() {
    LOGI("INFER", "tid=%u", static_cast<unsigned>(::gettid()));
    while (running_.load()) {
        InferenceInput ii;
        if (!ai_ring_->pop(ii, 200)) continue;

        DetectionResult det;
        det.pts_ms = ii.pts_ms;
        if (!detector_->detect(ii.rgb.data(), ii.w, ii.h, det)) continue;
        inferred_.fetch_add(1);

        // 疲劳分析
        AlarmEvent ev = analyzer_->process(det);
        cur_perclos_.store(analyzer_->context().perclos);
        cur_ear_.store(analyzer_->context().cur_ear);

        // 记录当前报警状态（供 display / stats 读取）
        cur_alarm_level_.store(static_cast<int>(ev.level));
        if (ev.level != AlarmLevel::kNone) {
            std::lock_guard<std::mutex> lk(alarm_reason_mtx_);
            cur_alarm_reason_ = ev.reason;
        } else {
            std::lock_guard<std::mutex> lk(alarm_reason_mtx_);
            if (!cur_alarm_reason_.empty()) cur_alarm_reason_.clear();
        }

        // 喂预警管理器（含冷却/TTS/录像触发）
        if (ev.level != AlarmLevel::kNone) {
            bool fired = alarm_->handle(ev);
            (void)fired;
            // 闭眼/疲劳报警：终端高亮提示（虚拟机验证用）
            if (ev.level >= AlarmLevel::kWarn) {
                LOGW("INFER", ">>> ALARM [%s] %s (EAR=%.2f PERCLOS=%.0f%%) <<<",
                     alarm_level_str(ev.level), ev.reason.c_str(),
                     cur_ear_.load(), cur_perclos_.load() * 100.0f);
            }
        }

        // 推送到预览显示（模型输入尺寸的画面，已叠加检测信息）
        if (display_) {
            std::string reason;
            {
                std::lock_guard<std::mutex> lk(alarm_reason_mtx_);
                reason = cur_alarm_reason_;
            }
            display_->update(ii.rgb.data(), ii.w, ii.h, det,
                             static_cast<AlarmLevel>(cur_alarm_level_.load()),
                             reason,
                             analyzer_->context().cur_ear_thresh);
        }
    }
    LOGI("INFER", "exited");
}

// ============== T5 视频编码消费 ==============
void DmsPipeline::encode_video_thread() {
    LOGI("VENC", "tid=%u", static_cast<unsigned>(::gettid()));
    while (running_.load()) {
        std::shared_ptr<VideoFrame> frame;
        if (!enc_video_ring_->pop(frame, 200)) continue;
        if (!frame) continue;
        venc_->push_frame(frame);
        // 编码通过回调完成，无需在此处理
    }
    LOGI("VENC", "exited");
}

// ============== T6 音频编码消费 ==============
void DmsPipeline::encode_audio_thread() {
    LOGI("AENC", "tid=%u", static_cast<unsigned>(::gettid()));
    while (running_.load()) {
        AudioFrame af;
        if (!raw_audio_ring_->pop(af, 200)) continue;
        aenc_->push_frame(af);
    }
    LOGI("AENC", "exited");
}

DmsPipeline::Stats DmsPipeline::stats() const {
    Stats s;
    if (v4l2_) {
        s.v_captured = v4l2_->captured_count();
        s.v_dropped  = v4l2_->dropped_count();
    }
    if (alsa_) {
        s.a_captured = alsa_->captured_count();
        s.a_dropped  = alsa_->dropped_count();
    }
    s.v_encoded = v_encoded_.load();
    s.a_encoded = a_encoded_.load();
    s.inferred  = inferred_.load();
    if (alarm_) {
        s.info_cnt   = alarm_->info_count();
        s.warn_cnt   = alarm_->warn_count();
        s.danger_cnt = alarm_->danger_count();
    }
    s.cur_perclos = cur_perclos_.load();
    s.cur_ear     = cur_ear_.load();
    return s;
}

}  // namespace dms
