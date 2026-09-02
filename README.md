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

## 设计模式

- **模板方法**：`FatigueAnalyzer` 定义"特征提取→时序累积→指标计算→策略汇总"标准流程
- **策略模式**：`IFatigueStrategy`（PERCLOS / 闭眼 / 哈欠 / 头垂 四个策略可插拔），`Detector`（Mock/RKNN）
- **单例**：`AesCrypto::instance()` 隐私加密模块
- **生产者-消费者**：`RingBuffer<T>` 串联各级线程

#结构

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

## 隐私合规

- 存证视频默认 AES-256-CTR 加密（`-e 1`），原始码流不出设备
- 密钥由设备唯一 ID 派生（KDF）
- 单例 `AesCrypto` 统一管理，避免多处持有
