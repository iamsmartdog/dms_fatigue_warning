# 车载驾驶员状态监测与疲劳预警系统（DMS）

> 平台：RK3588（Orange Pi 5B）+ USB 摄像头（LRCP20310_1080P）+ USB 麦克风
> 定位：嵌入式软件底层能力项目，AI 作为创新点

## 技术栈（Phase 1~6 全部实现）

| 模块 | 技术点 | Phase | 真实实现 | Stub（x86 可跑） |
|------|--------|-------|----------|------------------|
| 视频采集 | V4L2 ioctl/mmap/环形帧缓冲/PTS | 1 | ✅ | ✅ |
| 音频采集 | ALSA PCM/线性重采样 | 2 | ✅ `DMS_HAS_ALSA` | ✅ 静音发生器 |
| 视频编码 | RKMPP VPU H.265硬编码 | 3 | ✅ `DMS_HAS_RKMPP` | ✅ 占位 NALU |
| 音频编码 | AAC (FDK/FAAC) | 3 | ✅ `DMS_HAS_FDKAAC` | ✅ 占位帧 |
| 封装同步 | MP4/PTS 对齐 | 3 | ✅ `DMS_HAS_MP4V2` | ✅ 裸码流 .bin |
| 隐私加密 | AES-256-CTR 单例 | 3 | ✅ `DMS_HAS_OPENSSL` | ✅ XOR 占位 |
| 软件预处理 | MJPG 解码/resize/NV12/降噪 | 4 | ✅ `DMS_HAS_LIBJPEG` | ✅ 灰图占位 |
| 多线程 | 6 线程流水线 + 环形缓冲 | 4 | ✅ | ✅ |
| AI 推理 | RKNN INT8 人脸+关键点 | 5 | ✅ `DMS_HAS_RKNN` | ✅ Mock 周期眨眼 |
| 疲劳分析 | PERCLOS/EAR/MAR/姿态 | 6 | ✅ | ✅ |
| 预警录像 | 策略模式/事件触发录像 | 6 | ✅ | ✅ |

## 架构

```
   ┌─ T1 V4L2采集 ─→ raw_video_ring ─┐
   │                                  ├─→ T3 预处理 ─→ enc_video_ring ─→ T5 RKMPP编码 ─┐
   └─ T2 ALSA采集 ─→ raw_audio_ring ─┤                  │                                 │
                                      │                  └─→ ai_ring(降帧) ─→ T4 推理    │
                                      │                                       │          │
                                      └─→ T6 AAC编码 ─────────────────────────┐          │
                                                                              ↓          ↓
                                                                          EventRecorder ← AlarmManager
                                                                          (滚动缓冲+触发)
```

## 设计模式

- **模板方法**：`FatigueAnalyzer` 定义"特征提取→时序累积→指标计算→策略汇总"标准流程
- **策略模式**：`IFatigueStrategy`（PERCLOS / 闭眼 / 哈欠 / 头垂 四个策略可插拔），`Detector`（Mock/RKNN）
- **单例**：`AesCrypto::instance()` 隐私加密模块
- **生产者-消费者**：`RingBuffer<T>` 串联各级线程

## 编译

### x86 开发机（无任何硬件依赖，全部走 Stub，验证主流程）
```bash
cd build && cmake .. && make -j$(nproc)
```

### Orange Pi 5B（开启所有真实实现）
```bash
mkdir build-opi && cd build-opi
cmake .. \
    -DDMS_ENABLE_ALSA=ON \
    -DDMS_ENABLE_RKMPP=ON \
    -DDMS_ENABLE_FDKAAC=ON \
    -DDMS_ENABLE_MP4V2=ON \
    -DDMS_ENABLE_RKNN=ON \
    -DDMS_ENABLE_LIBJPEG=ON \
    -DDMS_ENABLE_OPENSSL=ON
make -j$(nproc)
```

## 运行

```bash
# 列出摄像头支持的格式
./dms_demo -d /dev/video0 -l

# 完整流水线（默认 Mock 检测器，开发机即可跑通）
./dms_demo -d /dev/video0 -a default -w 1280 -h 720 -f 30

# 上 RKNN 模型
./dms_demo -d /dev/video0 -m /usr/share/dms/face_landmark.rknn

# 不加密录像 / 调试保存原始帧
./dms_demo -d /dev/video0 -e 0 -s

# 详细日志
./dms_demo -d /dev/video0 -v
```

## 项目结构

```
dms_fatigue_warning/
├── src/
│   ├── capture/      # V4L2 视频采集 (Phase 1)
│   ├── audio/        # ALSA 音频采集 (Phase 2)
│   ├── preprocess/   # 软件预处理/ISP模拟 (Phase 4)
│   ├── encoder/      # RKMPP + AAC 编码 (Phase 3)
│   ├── muxer/        # MP4 封装 (Phase 3)
│   ├── crypto/       # AES 加密 单例 (Phase 3)
│   ├── ai/           # RKNN/Mock 检测 (Phase 5)
│   ├── alarm/        # 疲劳分析/预警/事件录像 (Phase 6)
│   ├── pipeline/     # 多线程流水线编排 (Phase 4)
│   ├── utils/        # 日志/环形缓冲/媒体类型
│   └── main.cpp
├── docs/design.md
└── CMakeLists.txt
```

## 硬件说明

- **USB 摄像头（LRCP20310_1080P）**：走 UVC 驱动，V4L2 采集正常，但**无法访问 RK3588 ISP**（HDR/3DNR/AE/AWB 锁定制），项目中以软件近似：对比度拉伸 + 时域降噪
- **MIPI 摄像头（可选）**：接入 CSI 接口才能做 ISP 调优；当前代码已为 MIPI 预留 NV12 直通路径
- **降帧策略**：采集 30fps，NPU 推理降到 15fps（`-i 15`），空闲时主线程可扩展休眠降功耗

## 隐私合规

- 存证视频默认 AES-256-CTR 加密（`-e 1`），原始码流不出设备
- 密钥由设备唯一 ID 派生（KDF）
- 单例 `AesCrypto` 统一管理，避免多处持有
