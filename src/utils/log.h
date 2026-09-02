#ifndef DMS_LOG_H
#define DMS_LOG_H

#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <sys/time.h>
#include <unistd.h>      // gettid()（需 _GNU_SOURCE）

namespace dms {

enum class LogLevel : uint8_t {
    kDebug = 0,
    kInfo  = 1,
    kWarn  = 2,
    kError = 3,
};

// 全局日志级别，可在 main 中调整
inline LogLevel& g_log_level() {
    static LogLevel level = LogLevel::kInfo;
    return level;
}

inline uint64_t now_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
}

inline void log_impl(LogLevel level, const char* tag, const char* fmt, ...) {
    if (static_cast<int>(level) < static_cast<int>(g_log_level())) return;

    const char* level_str[] = {"D", "I", "W", "E"};
    uint64_t ts = now_ms();
    fprintf(stderr, "[%llu][%s][%s] ", (unsigned long long)ts, level_str[static_cast<int>(level)], tag);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define LOGD(tag, ...) ::dms::log_impl(::dms::LogLevel::kDebug, tag, __VA_ARGS__)
#define LOGI(tag, ...) ::dms::log_impl(::dms::LogLevel::kInfo,  tag, __VA_ARGS__)
#define LOGW(tag, ...) ::dms::log_impl(::dms::LogLevel::kWarn,  tag, __VA_ARGS__)
#define LOGE(tag, ...) ::dms::log_impl(::dms::LogLevel::kError, tag, __VA_ARGS__)

}  // namespace dms

#endif  // DMS_LOG_H
