#include "alarm/event_recorder.h"

#include <chrono>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

#include "muxer/mp4_muxer.h"
#include "utils/log.h"

namespace dms {

EventRecorder::EventRecorder() = default;
EventRecorder::~EventRecorder() { flush(); }

bool EventRecorder::init() {
    // mkdir -p output_dir
    ::mkdir(cfg_.output_dir.c_str(), 0755);
    inited_ = true;
    LOGI(kTag, "event recorder ready, out=%s encrypt=%d",
         cfg_.output_dir.c_str(), (int)cfg_.encrypt);
    return true;
}

void EventRecorder::feed_packet(const EncodedPacket& pkt) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!inited_) return;

    // 1) 滚动缓冲维护：超容/超时则从头部弹出
    ring_.push_back(pkt);
    ring_last_pts_ = pkt.pts_ms;
    if (ring_first_pts_ == 0) ring_first_pts_ = pkt.pts_ms;
    while (ring_.size() > kRingMaxPackets ||
           (ring_last_pts_ > ring_first_pts_ &&
            ring_last_pts_ - ring_first_pts_ > cfg_.pre_roll_ms + 2000)) {
        ring_.pop_front();
        if (!ring_.empty()) ring_first_pts_ = ring_.front().pts_ms;
        else                ring_first_pts_ = 0;
    }

    // 2) 若有进行中的录像会话，直接喂入 muxer
    if (session_ && !session_->stopped) {
        session_->muxer->write_packet(pkt);
        if (pkt.pts_ms >= session_->stop_at_ms) {
            // post_roll 结束，关闭
            session_->muxer->close();
            LOGI(kTag, "record done: stop_at=%lu", (unsigned long)session_->stop_at_ms);
            session_.reset();
            active_.store(0);
        }
    }
}

void EventRecorder::on_alarm(const AlarmEvent& ev) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!inited_) return;
    if (static_cast<int>(ev.level) < static_cast<int>(cfg_.trigger_level)) return;

    uint64_t now = ev.pts_ms;
    uint64_t stop_at = now + cfg_.post_roll_ms;

    // 若已在录像中：延长 stop_at（避免连续事件切成多个文件）
    if (session_ && !session_->stopped) {
        if (stop_at > session_->stop_at_ms) {
            session_->stop_at_ms = stop_at;
            LOGI(kTag, "extend current record to %lu", (unsigned long)stop_at);
        }
        // 升级级别（只升不降）用于文件名是当前片段的级别
        if (static_cast<int>(ev.level) > static_cast<int>(session_->level))
            session_->level = ev.level;
        return;
    }

    // 新建会话：选文件名
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/dms_%lu_%s.mp4",
             cfg_.output_dir.c_str(),
             (unsigned long)(now / 1000),
             alarm_level_str(ev.level));
    MuxerConfig mcfg;
    mcfg.output_path = fname;
    mcfg.video_width = cfg_.enc_width;
    mcfg.video_height = cfg_.enc_height;
    mcfg.video_fps = cfg_.enc_fps;
    mcfg.audio_sample_rate = cfg_.audio_sample_rate;
    mcfg.audio_channels = cfg_.audio_channels;
    mcfg.encrypt = cfg_.encrypt;

    auto muxer = std::make_unique<Mp4Muxer>();
    if (!muxer->open(mcfg)) {
        LOGE(kTag, "open muxer failed: %s", fname);
        return;
    }

    auto sess = std::make_unique<Session>();
    sess->muxer = std::move(muxer);
    sess->start_ms = now;
    sess->stop_at_ms = stop_at;
    sess->level = ev.level;
    session_ = std::move(sess);
    active_.store(1);
    total_.fetch_add(1);
    LOGI(kTag, "EVENT RECORD start: %s (pre=%lu post=%lu)",
         fname, (unsigned long)cfg_.pre_roll_ms, (unsigned long)cfg_.post_roll_ms);

    // 回放滚动缓冲里 [pre_roll, now] 区间的包，作为"前置画面"
    uint64_t pre_start = (now > cfg_.pre_roll_ms) ? (now - cfg_.pre_roll_ms) : 0;
    int replayed = 0;
    for (const auto& p : ring_) {
        if (p.pts_ms >= pre_start && p.pts_ms <= now) {
            session_->muxer->write_packet(p);
            ++replayed;
        }
    }
    LOGI(kTag, "replayed %d buffered packets as pre-roll", replayed);
}

void EventRecorder::flush() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (session_) {
        if (!session_->stopped) session_->muxer->close();
        session_.reset();
        active_.store(0);
    }
}

}  // namespace dms
