#include "encoder/rkmpp_encoder.h"

#include <chrono>
#include <cstring>

#ifdef DMS_HAS_RKMPP
#include "rockchip/rk_mpi.h"
#include "rk_mpi_cmd.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#endif

#include "utils/log.h"

namespace dms {

// ============================================================
// 公共实现：缓冲、线程、回调（与是否有硬件无关）
// ============================================================

RkmppEncoder::RkmppEncoder() = default;

RkmppEncoder::~RkmppEncoder() {
    stop();
}

int RkmppEncoder::push_frame(std::shared_ptr<VideoFrame> frame) {
    if (!ring_) return -1;
    int dropped = ring_->push(frame);
    if (dropped > 0) dropped_.fetch_add(dropped);
    return dropped;
}

void RkmppEncoder::force_idr() {
    want_idr_.store(true);
    LOGI(kTag, "IDR requested");
}

bool RkmppEncoder::start() {
    if (running_.exchange(true)) return true;
    encoded_ = 0;
    dropped_ = 0;
    if (ring_) ring_->reset();
    thread_ = std::thread(&RkmppEncoder::encode_loop, this);
    LOGI(kTag, "encode thread started");
    return true;
}

void RkmppEncoder::stop() {
    if (!running_.exchange(false)) return;
    if (ring_) ring_->stop();
    if (thread_.joinable()) thread_.join();

#ifdef DMS_HAS_RKMPP
    if (mpp_api_ && mpp_ctx_) {
        MppApi* mpi = static_cast<MppApi*>(mpp_api_);
        mpi->reset(static_cast<MppCtx>(mpp_ctx_));
    }
    if (frame_grp_) {
        mpp_buffer_group_put(static_cast<MppBufferGroup>(frame_grp_));
        frame_grp_ = nullptr;
    }
    if (mpp_ctx_) {
        mpp_destroy(static_cast<MppCtx>(mpp_ctx_));
        mpp_ctx_ = nullptr;
    }
#endif
    LOGI(kTag, "encoder closed, encoded=%lu dropped=%lu",
         encoded_.load(), dropped_.load());
}

void RkmppEncoder::encode_loop() {
    LOGI(kTag, "encode_loop tid=%u", static_cast<unsigned>(::gettid()));
    while (running_.load()) {
        std::shared_ptr<VideoFrame> frame;
        if (!ring_->pop(frame, 100)) continue;  // 100ms 心跳，便于退出
        if (!frame || !frame->is_valid) continue;
        encode_one(*frame);
    }
    LOGI(kTag, "encode_loop exited");
}

// ============================================================
// 硬件实现 / Stub
// ============================================================

#ifdef DMS_HAS_RKMPP

bool RkmppEncoder::open(const EncoderConfig& cfg) {
    cfg_ = cfg;

    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;

    if (mpp_create(&ctx, &mpi) != MPP_OK) {
        LOGE(kTag, "mpp_create failed");
        return false;
    }
    if (mpp_init(ctx, MPP_CTX_ENC, MPP_VIDEO_CodingHEVC) != MPP_OK) {
        LOGE(kTag, "mpp_init HEVC encoder failed");
        mpp_destroy(ctx);
        return false;
    }
    mpp_ctx_ = ctx;
    mpp_api_ = mpi;

    // 配置编码参数
    MppEncCfg cfg_enc;
    mpp_enc_cfg_init(&cfg_enc);

    // prep（输入格式）
    mpp_enc_cfg_set_fmt(cfg_enc, cfg.input_fmt == PixelFormat::kNv12
                                     ? MPP_FMT_YUV420SP
                                     : MPP_FMT_YUV422_YUYV);
    mpp_enc_cfg_set_prep(cfg_enc, cfg_.width, cfg_.height,
                         MPP_HOR_STRIDE(cfg_.width), MPP_VIR_STRIDE(cfg_.height));

    // rc（码率控制）：CBR，适合固定码率录像
    mpp_enc_cfg_set_rc_mode(cfg_enc, MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_rc_fps(cfg_enc, cfg_.fps, 1);
    mpp_enc_cfg_set_rc_gop(cfg_enc, cfg_.gop);
    mpp_enc_cfg_set_rc_bitrate(cfg_enc, cfg_.bitrate, cfg_.bitrate, cfg_.bitrate);
    mpp_enc_cfg_set_rc_qp(cfg_enc, cfg_.qp_init, cfg_.qp_min, cfg_.qp_max);

    // h265 专用
    mpp_enc_cfg_set_codec(cfg_enc, MPP_VIDEO_CodingHEVC);

    if (mpi->control(ctx, MPP_ENC_SET_CFG, cfg_enc) != MPP_OK) {
        LOGE(kTag, "MPP_ENC_SET_CFG failed");
        mpp_enc_cfg_deinit(cfg_enc);
        return false;
    }
    mpp_enc_cfg_deinit(cfg_enc);

    // 帧输入 buffer group
    if (mpp_buffer_group_get_internal(&frame_grp_, MPP_BUFFER_TYPE_ION) != MPP_OK) {
        LOGE(kTag, "frame buffer group get failed");
        return false;
    }

    ring_ = std::make_unique<RingBuffer<std::shared_ptr<VideoFrame>>>(cfg_.ring_capacity);
    LOGI(kTag, "RKMPP HEVC encoder ready: %ux%u@%u gop=%u br=%u",
         cfg_.width, cfg_.height, cfg_.fps, cfg_.gop, cfg_.bitrate);
    return true;
}

bool RkmppEncoder::encode_one(const VideoFrame& frame) {
    MppApi* mpi = static_cast<MppApi*>(mpp_api_);
    MppCtx  ctx = static_cast<MppCtx>(mpp_ctx_);

    // 强制 IDR
    if (want_idr_.exchange(false)) {
        mpi->control(ctx, MPP_ENC_SET_IDR_FRAME, nullptr);
    }

    // 1) 帧 input
    MppFrame mframe = nullptr;
    mpp_frame_init(&mframe);
    mpp_frame_set_width(mframe, cfg_.width);
    mpp_frame_set_height(mframe, cfg_.height);
    mpp_frame_set_hor_stride(mframe, MPP_HOR_STRIDE(cfg_.width));
    mpp_frame_set_ver_stride(mframe, MPP_VIR_STRIDE(cfg_.height));
    mpp_frame_set_fmt(mframe, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(mframe, frame.pts_ms);

    // 把 NV12 数据拷入 MppBuffer（USB 摄像头场景无 DMA-BUF，必须拷一次）
    MppBuffer mbuf = nullptr;
    size_t frame_size = static_cast<size_t>(cfg_.width) * cfg_.height * 3 / 2;
    mpp_buffer_get(frame_grp_, &mbuf, frame_size);
    std::memcpy(mpp_buffer_get_ptr(mbuf), frame.data,
                std::min(frame.size, frame_size));
    mpp_frame_set_buffer(mframe, mbuf);
    mpp_buffer_put(mbuf);

    if (mpi->encode_put_frame(ctx, mframe) != MPP_OK) {
        LOGE(kTag, "encode_put_frame failed");
        mpp_frame_deinit(&mframe);
        return false;
    }
    mpp_frame_deinit(&mframe);

    // 2) 码流 output
    MppPacket mpkt = nullptr;
    if (mpi->encode_get_packet(ctx, &mpkt) != MPP_OK) {
        LOGE(kTag, "encode_get_packet failed");
        return false;
    }

    EncodedPacket pkt;
    pkt.type = MediaType::kVideo;
    pkt.pts_ms = frame.pts_ms;
    pkt.dts_ms = frame.pts_ms;
    // RK 码流包首字节高位标记 keyframe（mpp_packet_is_partition flag）
    pkt.keyframe = (mpp_packet_get_flag(mpkt) & 0x02) != 0;

    void* ptr = mpp_packet_get_pos(mpkt);
    size_t len = mpp_packet_get_length(mpkt);
    pkt.data.assign(static_cast<uint8_t*>(ptr),
                    static_cast<uint8_t*>(ptr) + len);

    if (on_encoded_) on_encoded_(pkt);
    encoded_.fetch_add(1);

    mpp_packet_deinit(&mpkt);
    return true;
}

#else  // !DMS_HAS_RKMPP

// Stub：无 RKMPP 时，把输入帧"伪装"成码流包（仅占位字节），
// 让 Muxer / 流水线在开发机上跑通
bool RkmppEncoder::open(const EncoderConfig& cfg) {
    cfg_ = cfg;
    ring_ = std::make_unique<RingBuffer<std::shared_ptr<VideoFrame>>>(cfg_.ring_capacity);
    LOGW(kTag, "[STUB] RKMPP not linked, mock HEVC encoder ready");
    return true;
}

bool RkmppEncoder::encode_one(const VideoFrame& frame) {
    EncodedPacket pkt;
    pkt.type = MediaType::kVideo;
    pkt.pts_ms = frame.pts_ms;
    pkt.dts_ms = frame.pts_ms;
    // 模拟 H.265 VPS/SPS/PPS/IDR 占位（实际是 RAW 帧字节前缀）
    static const uint8_t kFakeNalu[] = {0x00, 0x00, 0x00, 0x01, 0x40, 0x01};
    pkt.data.assign(kFakeNalu, kFakeNalu + sizeof(kFakeNalu));
    // 追加若干原始字节作为"payload"，便于验证大小
    pkt.data.push_back(static_cast<uint8_t>(frame.seq & 0xFF));
    pkt.keyframe = want_idr_.exchange(false) || (frame.seq % cfg_.gop == 0);

    if (on_encoded_) on_encoded_(pkt);
    encoded_.fetch_add(1);
    return true;
}

#endif  // DMS_HAS_RKMPP

}  // namespace dms
