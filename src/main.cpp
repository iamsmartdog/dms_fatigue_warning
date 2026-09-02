#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "pipeline/dms_pipeline.h"
#include "utils/log.h"

static std::atomic<bool> g_exit{false};

static void on_signal(int sig) {
    g_exit.store(true);
    fprintf(stderr, "\ncaught signal %d, exiting...\n", sig);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -d <dev>    video device (default /dev/video0)\n"
        "  -a <dev>    audio device (default default)\n"
        "  -w <num>    width  (default 1280)\n"
        "  -h <num>    height (default 720)\n"
        "  -f <num>    fps    (default 30)\n"
        "  -t <type>   detector: mock | cv | rknn (default mock)\n"
        "  -m <path>   model path (rknn) / haarcascade dir (cv)\n"
        "  -r <dir>    record output dir (default ./records)\n"
        "  -b <num>    video bitrate bps (default 2000000)\n"
        "  -i <num>    inference fps (default 15)\n"
        "  -e <0|1>    encrypt record (default 1)\n"
        "  -p          enable preview window (imshow, dev/vm only)\n"
        "  -v          verbose (debug log)\n"
        "  -l          list formats and exit (Phase1 debug)\n"
        "  -s          save first 3 raw frames (debug)\n"
        "  --pfld <path>  PFLD 106-point landmark model (yunet only)\n",
        prog);
}

int main(int argc, char* argv[]) {
    dms::g_log_level() = dms::LogLevel::kInfo;
    ::signal(SIGINT, on_signal);
    ::signal(SIGTERM, on_signal);

    dms::DmsPipeline::Config cfg;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* d) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string(d);
        };
        if      (arg == "-d") cfg.video_device = next("/dev/video0");
        else if (arg == "-a") cfg.audio_device = next("default");
        else if (arg == "-w") cfg.video_width  = std::atoi(argv[++i]);
        else if (arg == "-h") cfg.video_height = std::atoi(argv[++i]);
        else if (arg == "-f") cfg.video_fps    = std::atoi(argv[++i]);
        else if (arg == "-t") cfg.detector_type = argv[++i];
        else if (arg == "-m") cfg.model_path   = argv[++i];
        else if (arg == "-r") cfg.record_dir   = argv[++i];
        else if (arg == "-b") cfg.enc_bitrate  = std::atoi(argv[++i]);
        else if (arg == "-i") cfg.inference_fps = std::atoi(argv[++i]);
        else if (arg == "-e") cfg.encrypt = std::atoi(argv[++i]) != 0;
        else if (arg == "-p") cfg.enable_preview = true;
        else if (arg == "-v") dms::g_log_level() = dms::LogLevel::kDebug;
        else if (arg == "-s") cfg.stub_dump_first_frames = true;
        else if (arg == "-l") list_only = true;
        else if (arg == "--pfld") cfg.pfld_model_path = next("");
        else { print_usage(argv[0]); return 0; }
    }

    LOGI("MAIN", "=== DMS Fatigue Warning System ===");
    LOGI("MAIN", "video=%s %ux%u@%u audio=%s detector=%s model=%s preview=%d",
         cfg.video_device.c_str(), cfg.video_width, cfg.video_height, cfg.video_fps,
         cfg.audio_device.c_str(), cfg.detector_type.c_str(),
         cfg.model_path.empty() ? "<auto>" : cfg.model_path.c_str(),
         cfg.enable_preview ? 1 : 0);

    // -l 调试路径：仅列出摄像头格式
    if (list_only) {
        dms::V4l2Capture cap;
        if (cap.open(cfg.video_device)) cap.list_formats();
        return 0;
    }

    dms::DmsPipeline pipeline;
    if (!pipeline.init(cfg)) {
        LOGE("MAIN", "pipeline init failed");
        return 1;
    }
    if (!pipeline.start()) {
        LOGE("MAIN", "pipeline start failed");
        return 1;
    }

    // 主线程：每秒打印统计，直到收到信号
    while (!g_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        auto s = pipeline.stats();
        LOGI("MAIN",
             "[stats] v_cap=%lu(drop %lu) a_cap=%lu(drop %lu) "
             "v_enc=%lu a_enc=%lu infer=%lu | "
             "alarm I/W/D=%lu/%lu/%lu | EAR=%.2f PERCLOS=%.0f%%",
             (unsigned long)s.v_captured, (unsigned long)s.v_dropped,
             (unsigned long)s.a_captured, (unsigned long)s.a_dropped,
             (unsigned long)s.v_encoded, (unsigned long)s.a_encoded,
             (unsigned long)s.inferred,
             (unsigned long)s.info_cnt, (unsigned long)s.warn_cnt,
             (unsigned long)s.danger_cnt,
             s.cur_ear, s.cur_perclos * 100.0f);
    }

    LOGI("MAIN", "stopping pipeline...");
    pipeline.stop();
    LOGI("MAIN", "=== DMS exit ===");
    return 0;
}
