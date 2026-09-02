cmake_minimum_required(VERSION 3.15)
project(dms_fatigue_warning VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -g)

# _GNU_SOURCE：使 gettid() 等扩展函数可用（glibc >= 2.30）
add_compile_definitions(_GNU_SOURCE)

# 架构检测（区分 x86_64 开发机与 aarch64 目标板）
message(STATUS "Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# ====================================================================
# 平台能力开关：开发机默认全部 OFF，所有模块走 Stub；上板时手动开启
# ====================================================================
option(DMS_ENABLE_ALSA    "Enable real ALSA audio capture"           OFF)
option(DMS_ENABLE_RKMPP   "Enable RKMPP hardware video encoder"      OFF)
option(DMS_ENABLE_FDKAAC  "Enable FDK-AAC encoder"                   OFF)
option(DMS_ENABLE_FAAC    "Enable FAAC encoder"                      OFF)
option(DMS_ENABLE_MP4V2   "Enable MP4V2 muxer (real MP4 output)"     OFF)
option(DMS_ENABLE_RKNN    "Enable RKNN NPU inference"                OFF)
option(DMS_ENABLE_LIBJPEG "Enable libjpeg-turbo MJPG decode"         OFF)
option(DMS_ENABLE_OPENSSL "Enable OpenSSL (AES + SHA KDF)"           OFF)
option(DMS_ENABLE_OPENCV  "Enable OpenCV (YuNet detector + preview)"  OFF)

# 把开关转成 -D 宏，供 #ifdef 分支选择真实实现 / stub
add_compile_definitions(
    $<$<BOOL:${DMS_ENABLE_ALSA}>:DMS_HAS_ALSA>
    $<$<BOOL:${DMS_ENABLE_RKMPP}>:DMS_HAS_RKMPP>
    $<$<BOOL:${DMS_ENABLE_FDKAAC}>:DMS_HAS_FDKAAC>
    $<$<BOOL:${DMS_ENABLE_FAAC}>:DMS_HAS_FAAC>
    $<$<BOOL:${DMS_ENABLE_MP4V2}>:DMS_HAS_MP4V2>
    $<$<BOOL:${DMS_ENABLE_RKNN}>:DMS_HAS_RKNN>
    $<$<BOOL:${DMS_ENABLE_LIBJPEG}>:DMS_HAS_LIBJPEG>
    $<$<BOOL:${DMS_ENABLE_OPENSSL}>:DMS_HAS_OPENSSL>
    $<$<BOOL:${DMS_ENABLE_OPENCV}>:DMS_HAS_OPENCV>
)

# ====================================================================
# 核心库：DMS 各模块（Phase 1~6 全部编入，按宏选择实现）
# ====================================================================
add_library(dms_core STATIC
    # Phase 1 视频采集
    src/capture/v4l2_capture.cpp
    # Phase 2 音频采集
    src/audio/alsa_capture.cpp
    # Phase 3 编码 + 封装 + 加密
    src/encoder/rkmpp_encoder.cpp
    src/encoder/aac_encoder.cpp
    src/muxer/mp4_muxer.cpp
    src/crypto/aes_crypto.cpp
    # Phase 4 预处理
    src/preprocess/image_preprocess.cpp
    # Phase 5 AI 推理
    src/ai/detector.cpp
    src/ai/mock_detector.cpp
    src/ai/rknn_detector.cpp
    src/ai/cv_detector.cpp
    src/ai/yunet_detector.cpp
    src/ai/pfld_detector.cpp
    # Phase 6 疲劳分析 + 预警 + 事件录像
    src/alarm/fatigue_analyzer.cpp
    src/alarm/alarm_manager.cpp
    src/alarm/event_recorder.cpp
    # 显示预览（OpenCV imshow）
    src/display/preview_display.cpp
)
target_include_directories(dms_core PUBLIC src)

find_package(Threads REQUIRED)
target_link_libraries(dms_core PUBLIC Threads::Threads)

# 平台库链接（按开关查找/链接）
if(DMS_ENABLE_ALSA)
    find_package(ALSA REQUIRED)
    target_link_libraries(dms_core PUBLIC ${ALSA_LIBRARIES})
endif()
if(DMS_ENABLE_RKMPP)
    target_include_directories(dms_core PRIVATE /usr/local/include/rockchip)
    target_link_libraries(dms_core PRIVATE rockchip_mpp)
endif()
if(DMS_ENABLE_FDKAAC)
    target_link_libraries(dms_core PRIVATE fdk-aac)
endif()
if(DMS_ENABLE_FAAC)
    target_link_libraries(dms_core PRIVATE faac)
endif()
if(DMS_ENABLE_MP4V2)
    target_link_libraries(dms_core PRIVATE mp4v2)
endif()
if(DMS_ENABLE_RKNN)
    target_include_directories(dms_core PRIVATE /usr/local/include)
    target_link_libraries(dms_core PRIVATE rknn_api)
endif()
if(DMS_ENABLE_LIBJPEG)
    find_package(JPEG REQUIRED)
    target_link_libraries(dms_core PRIVATE ${JPEG_LIBRARIES})
endif()
if(DMS_ENABLE_OPENSSL)
    target_link_libraries(dms_core PRIVATE ssl crypto)
endif()
if(DMS_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED)
    target_link_libraries(dms_core PRIVATE ${OpenCV_LIBS})
    target_include_directories(dms_core PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# ====================================================================
# 流水线可执行：包含 main + pipeline 编排
# ====================================================================
add_executable(dms_demo
    src/main.cpp
    src/pipeline/pipeline_stage.h    # 头文件 only（模板）
    src/pipeline/dms_pipeline.cpp
)
target_link_libraries(dms_demo PRIVATE dms_core)

# ====================================================================
# 安装
# ====================================================================
install(TARGETS dms_demo RUNTIME DESTINATION bin)
install(DIRECTORY src/ DESTINATION include/dms FILES_MATCHING PATTERN "*.h")
