# DMS 架构设计

## 数据流（已实现）

```
┌──────────┐ raw_video_ring ┌──────────┐ enc_video_ring ┌──────────┐
│ T1 V4L2  │───────────────>│ T3 预处理│───────────────>│ T5 RKMPP │──┐
│ 采集     │                │ (双输出) │                │  编码    │  │
└──────────┘                └────┬─────┘                └──────────┘  │
                                 │ ai_ring (降帧)                     │
                                 ▼                                    │
┌──────────┐ raw_audio_ring ┌──────────┐                              │
│ T2 ALSA  │───────────────>│ T6 AAC   │──────────────────────────────┤
│ 采集     │                │  编码    │                              │
└──────────┘                └──────────┘                              ▼
                                                                 ┌──────────┐
                                 ┌──────────┐    AlarmEvent        │ Event    │
                                 │ T4 推理  │──────┬──────────────>│ Recorder │
                            ┌───>│ +疲劳分析│      │                │ (滚动缓冲│
                            │    └──────────┘      ▼                │  +触发) │
                            │                   ┌──────────┐        └──────────┘
                            └───────────────────│ Alarm    │              │
                                DetectionResult │ Manager  │              ▼
                                                 │ (冷却/TTS)│         records/
                                                 └──────────┘       *.mp4 (加密)
```

## 线程模型（6 个工作线程 + 主线程）

| 线程 | 类 | 职责 | 优先级 |
|------|----|------|--------|
| T1 vcap  | V4l2Capture | DQBUF → 深拷贝 → raw_video_ring | 高 |
| T2 acap  | AlsaCapture | PCM readi → 重采样 → raw_audio_ring | 高 |
| T3 prep  | ImagePreprocess | 解码/resize/NV12+RGB/降噪（双输出） | 中 |
| T4 infer | Detector+FatigueAnalyzer+AlarmManager | NPU 推理 → PERCLOS → 预警 | 中 |
| T5 venc  | RkmppEncoder | H.265 编码 → 回调 recorder | 中 |
| T6 aenc  | AacEncoder | AAC 编码 → 回调 recorder | 中 |
| 主线程   | DmsPipeline::stats | 监控/信号处理 | 低 |

## 帧缓冲策略

- **采集→处理**：`RingBuffer<shared_ptr<VideoFrame>>`，满则丢老帧（实时优先）
- **处理→编码**：同上，编码前 `enc_video_ring_`
- **处理→推理**：`ai_ring_`，容量 4，且每 N 帧才入队（降帧）
- **编码→录像**：编码器内部回调直接 `recorder_->feed_packet()`，无中间缓冲

## 音视频同步（PTS）

- V4L2 帧时间戳（`buf.timestamp`）→ `pts_ms` 毫秒
- ALSA：基于输出样本数 × 1000 / sample_rate + start_ms，与视频同一系统时钟域
- AAC 内部按 1024 samples 分块，块 PTS 由累积样本数推导
- MP4 封装：各 sample 写入时换算到该轨 timescale；moov 在末尾写

## 设计模式

### 模板方法（`FatigueAnalyzer`）
```cpp
AlarmEvent process(det) {
    extract_features(det);    // Step1: 提取 EAR/MAR/姿态
    accumulate_timeline();    // Step2: 时序累积/闭眼计时
    compute_metrics();        // Step3: PERCLOS / 眨眼频率
    return aggregate(...);    // Step4: 多策略汇总
}
```

### 策略模式
- `IFatigueStrategy`：4 个内置策略（PERCLOS / EyeClosure / Yawn / HeadDrop），可 `add_strategy()` 扩展
- `Detector`：`MockDetector`（开发机）/ `RknnDetector`（板子），运行时按 `-m` 选择

### 单例
- `AesCrypto::instance()`：全局唯一密钥管理，CTR 流式加密

## Phase 进度

- [x] Phase 1: V4L2 采集 + 环形缓冲
- [x] Phase 2: ALSA 音频采集 + 重采样
- [x] Phase 3: RKMPP 编码 + AAC + MP4 + AES
- [x] Phase 4: 多线程流水线 + 软件预处理
- [x] Phase 5: RKNN AI 推理 + Mock
- [x] Phase 6: 疲劳分析 + 策略预警 + 事件录像

## USB 摄像头 vs MIPI 摄像头

由于当前硬件为 USB UVC 摄像头，**无法访问 RK3588 ISP**（HDR/3DNR/AE/AWB 锁定制），项目以软件方案近似：
- 对比度拉伸 → 近似 HDR 宽动态
- 时域 IIR 降噪 → 近似 3DNR

代码中预留了 MIPI/NV12 直通路径（`PixelFormat::kNv12`），后续接入 MIPI 摄像头时只需：
1. `v4l2_capture` 改为 `MEDIA_BUS_FMT_*` + NV12 格式
2. 加入 `rkisp` 节点配置 ISP 参数
3. 启用 DMA-BUF fd 传递（`VIDIOC_EXPBUF` → 传给 MPP / RKNN，真正零拷贝）

## 降帧与功耗优化

- 采集 30fps / 推理 15fps：`preprocess_thread` 中按 `video_fps/inference_fps - 1` 跳帧
- 空闲休眠（待扩展）：长时间无人脸时降低采集帧率，NPU 进入低功耗模式
