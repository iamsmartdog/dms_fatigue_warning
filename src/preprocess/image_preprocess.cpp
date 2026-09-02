#include "preprocess/image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef DMS_HAS_LIBJPEG
#include <jpeglib.h>
#endif

#include "utils/log.h"

namespace dms {

// ---------------- 颜色空间转换 ----------------

void ImagePreprocess::yuyv_to_rgb(const uint8_t* yuyv, uint32_t w, uint32_t h,
                                  std::vector<uint8_t>& rgb) {
    rgb.resize(static_cast<size_t>(w) * h * 3);
    for (uint32_t i = 0; i < w * h / 2; ++i) {
        // YUYV: Y0 U0 Y1 V0
        int y0 = yuyv[i * 4 + 0];
        int u  = yuyv[i * 4 + 1] - 128;
        int y1 = yuyv[i * 4 + 2];
        int v  = yuyv[i * 4 + 3] - 128;

        auto yuv2rgb = [](int y, int u, int v) {
            int r = y + static_cast<int>(1.402f * v);
            int g = y - static_cast<int>(0.344f * u) - static_cast<int>(0.714f * v);
            int b = y + static_cast<int>(1.772f * u);
            r = std::clamp(r, 0, 255);
            g = std::clamp(g, 0, 255);
            b = std::clamp(b, 0, 255);
            return std::make_tuple(r, g, b);
        };
        auto [r0, g0, b0] = yuv2rgb(y0, u, v);
        auto [r1, g1, b1] = yuv2rgb(y1, u, v);
        size_t o0 = static_cast<size_t>(i * 2) * 3;
        rgb[o0 + 0] = r0; rgb[o0 + 1] = g0; rgb[o0 + 2] = b0;
        rgb[o0 + 3] = r1; rgb[o0 + 4] = g1; rgb[o0 + 5] = b1;
    }
}

void ImagePreprocess::rgb_to_nv12(const uint8_t* rgb, uint32_t w, uint32_t h,
                                  std::vector<uint8_t>& nv12) {
    nv12.resize(static_cast<size_t>(w) * h * 3 / 2);
    uint8_t* y_plane = nv12.data();
    uint8_t* uv_plane = nv12.data() + static_cast<size_t>(w) * h;

    for (uint32_t j = 0; j < h; ++j) {
        for (uint32_t i = 0; i < w; ++i) {
            const uint8_t* p = rgb + (static_cast<size_t>(j) * w + i) * 3;
            int r = p[0], g = p[1], b = p[2];
            // BT.601
            int y = static_cast<int>(0.257f * r + 0.504f * g + 0.098f * b + 16);
            y_plane[j * w + i] = static_cast<uint8_t>(std::clamp(y, 0, 255));

            // UV 每 2x2 像素取一次
            if ((j & 1) == 0 && (i & 1) == 0) {
                const uint8_t* p2 = rgb + (static_cast<size_t>(j) * w + i + 1) * 3;
                const uint8_t* p3 = rgb + (static_cast<size_t>(j + 1) * w + i) * 3;
                const uint8_t* p4 = rgb + (static_cast<size_t>(j + 1) * w + i + 1) * 3;
                int rr = (r + p2[0] + p3[0] + p4[0]) / 4;
                int gg = (g + p2[1] + p3[1] + p4[1]) / 4;
                int bb = (b + p2[2] + p3[2] + p4[2]) / 4;
                int u = static_cast<int>(-0.148f * rr - 0.291f * gg + 0.439f * bb + 128);
                int v = static_cast<int>(0.439f * rr - 0.368f * gg - 0.071f * bb + 128);
                size_t uv_idx = (j / 2) * w + (i / 2) * 2;
                uv_plane[uv_idx + 0] = static_cast<uint8_t>(std::clamp(u, 0, 255));
                uv_plane[uv_idx + 1] = static_cast<uint8_t>(std::clamp(v, 0, 255));
            }
        }
    }
}

// ---------------- resize（双线性） ----------------
void ImagePreprocess::resize_bilinear(const uint8_t* src, uint32_t sw, uint32_t sh,
                                      uint8_t* dst, uint32_t dw, uint32_t dh,
                                      uint32_t channels) {
    if (sw == 0 || sh == 0 || dw == 0 || dh == 0) return;
    const float sx = static_cast<float>(sw) / dw;
    const float sy = static_cast<float>(sh) / dh;

    for (uint32_t y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * sy - 0.5f;
        uint32_t sy0 = std::clamp(static_cast<int>(fy), 0, static_cast<int>(sh) - 1);
        uint32_t sy1 = std::min(sy0 + 1, sh - 1);
        float wy = fy - sy0;
        if (wy < 0) wy = 0;
        for (uint32_t x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f;
            uint32_t sx0 = std::clamp(static_cast<int>(fx), 0, static_cast<int>(sw) - 1);
            uint32_t sx1 = std::min(sx0 + 1, sw - 1);
            float wx = fx - sx0;
            if (wx < 0) wx = 0;
            for (uint32_t c = 0; c < channels; ++c) {
                float v00 = src[(sy0 * sw + sx0) * channels + c];
                float v01 = src[(sy0 * sw + sx1) * channels + c];
                float v10 = src[(sy1 * sw + sx0) * channels + c];
                float v11 = src[(sy1 * sw + sx1) * channels + c];
                float v0 = v00 * (1 - wx) + v01 * wx;
                float v1 = v10 * (1 - wx) + v11 * wx;
                dst[(y * dw + x) * channels + c] =
                    static_cast<uint8_t>(v0 * (1 - wy) + v1 * wy);
            }
        }
    }
}

// ---------------- 增强 ----------------
uint8_t ImagePreprocess::compute_brightness(const std::vector<uint8_t>& rgb,
                                            uint32_t w, uint32_t h) {
    if (rgb.empty()) return 128;
    size_t total = static_cast<size_t>(w) * h;
    if (total == 0) return 128;
    // 间隔采样提升速度（每 4 像素取 1）
    size_t step = 4;
    uint64_t sum = 0;
    size_t cnt = 0;
    for (size_t i = 0; i < rgb.size(); i += step * 3) {
        uint8_t r = rgb[i + 0];
        uint8_t g = rgb[i + 1];
        uint8_t b = rgb[i + 2];
        // 感知亮度 (BT.709)
        sum += static_cast<uint64_t>(0.2126f * r + 0.7152f * g + 0.0722f * b);
        ++cnt;
    }
    if (cnt == 0) return 128;
    uint8_t brightness = static_cast<uint8_t>(sum / cnt);
    // 平滑：与上一帧加权，避免闪变
    last_brightness_ = static_cast<uint8_t>(
        last_brightness_ * 0.7f + brightness * 0.3f);
    return last_brightness_;
}

void ImagePreprocess::contrast_enhance(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h) {
    if (rgb.empty()) return;
    // 改进版对比度增强：更温和，避免过度拉伸
    // 统计直方图，用 2% ~ 98% 分位点（更宽松）
    std::vector<size_t> hist(256, 0);
    for (uint8_t v : rgb) ++hist[v];
    size_t total = static_cast<size_t>(w) * h * 3;  // RGB 三通道总像素
    if (total == 0) return;
    
    // 累积直方图
    std::vector<size_t> cdf(256, 0);
    size_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[i];
        cdf[i] = acc;
    }
    
    // 找 2% 和 98% 分位点（比 1-99 更保守）
    size_t lo_cut = total * 2 / 100;
    size_t hi_cut = total * 98 / 100;
    uint8_t mn = 0, mx = 255;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > lo_cut) { mn = static_cast<uint8_t>(i); break; }
    }
    for (int i = 255; i >= 0; --i) {
        if (cdf[i] < hi_cut) { mx = static_cast<uint8_t>(i); break; }
    }
    
    // 如果动态范围太小（mn ~ mx < 50），放弃拉伸，避免过度增强
    if (mx - mn < 50) return;
    
    // 应用分段线性映射（避免直接拉满）
    // 把 [mn, mx] 映射到 [0, 255]，但限制增益到 1.5 倍
    const float range = static_cast<float>(mx - mn);
    const float max_gain = 1.5f;
    const float target_range = std::min(255.0f, range * max_gain);
    const float scale = target_range / range;
    
    for (auto& v : rgb) {
        if (v < mn) continue;  // 保持黑色区域
        if (v > mx) {
            v = static_cast<uint8_t>(std::min(255.0f, mn + range + (v - mx) * 0.5f));
        } else {
            float mapped = (v - mn) * scale;
            v = static_cast<uint8_t>(std::clamp(mapped, 0.0f, 255.0f));
        }
    }
}

void ImagePreprocess::clahe_enhance(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h) {
    if (rgb.empty() || w == 0 || h == 0) return;
    // 简化版 CLAHE 风格增强：在亮度域（Y）做局部直方图均衡
    // 先转灰度近似，分块统计，再映射回 RGB
    // 这里使用更简单的方法：对每个通道分别做自适应直方图均衡
    // 限制对比度，防止过曝
    
    // 3 通道各自独立均衡（保持色相）
    for (int c = 0; c < 3; ++c) {
        std::vector<size_t> hist(256, 0);
        for (size_t i = static_cast<size_t>(c); i < rgb.size(); i += 3) ++hist[rgb[i]];
        size_t total = static_cast<size_t>(w) * h;
        if (total == 0) continue;
        
        // CLAHE 限幅：限制直方图峰值
        const float clip_limit = 2.0f;
        size_t clip = static_cast<size_t>(clip_limit * total / 256);
        size_t excess = 0;
        std::vector<size_t> clipped(256, 0);
        for (int i = 0; i < 256; ++i) {
            if (hist[i] > clip) {
                excess += hist[i] - clip;
                clipped[i] = clip;
            } else {
                clipped[i] = hist[i];
            }
        }
        // 重新分配 excess
        size_t add = excess / 256;
        size_t rem = excess % 256;
        for (int i = 0; i < 256; ++i) {
            clipped[i] += add;
            if (i < static_cast<int>(rem)) clipped[i]++;
        }
        
        // CDF 映射
        std::vector<uint8_t> lut(256);
        size_t acc = 0;
        for (int i = 0; i < 256; ++i) {
            acc += clipped[i];
            lut[i] = static_cast<uint8_t>(255 * acc / total);
        }
        
        // 应用映射
        for (size_t i = static_cast<size_t>(c); i < rgb.size(); i += 3) {
            rgb[i] = lut[rgb[i]];
        }
    }
}

void ImagePreprocess::temporal_denoise(std::vector<uint8_t>& rgb, uint32_t w, uint32_t h) {
    if (!cfg_.enable_denoise) return;
    if (!has_prev_ || prev_frame_.size() != rgb.size()) {
        prev_frame_ = rgb;
        has_prev_ = true;
        return;
    }
    // IIR：out = alpha * prev + (1-alpha) * cur
    const float a = cfg_.denoise_alpha;
    const float b = 1.0f - a;
    for (size_t i = 0; i < rgb.size(); ++i) {
        float v = prev_frame_[i] * a + rgb[i] * b;
        prev_frame_[i] = static_cast<uint8_t>(v);
        rgb[i] = prev_frame_[i];
    }
}

// ---------------- MJPG 解码 ----------------
bool ImagePreprocess::decode_mjpg_to_rgb(const uint8_t* jpg, size_t size,
                                         std::vector<uint8_t>& rgb,
                                         uint32_t& w, uint32_t& h) {
#ifdef DMS_HAS_LIBJPEG
    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<uint8_t*>(jpg), size);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    w = cinfo.output_width;
    h = cinfo.output_height;
    rgb.resize(static_cast<size_t>(w) * h * 3);
    while (cinfo.output_scanline < h) {
        uint8_t* row = rgb.data() + static_cast<size_t>(cinfo.output_scanline) * w * 3;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
#else
    // Stub：没法解码 MJPG，填充 1x1 灰，方便流水线跑通
    (void)jpg; (void)size;
    rgb.assign(3 * 4, 128);  // 2x2 灰
    w = 2; h = 2;
    LOGW("PREP", "[STUB] libjpeg not linked, MJPG decode mocked");
    return true;
#endif
}

// ---------------- 一站式 ----------------
ImagePreprocess::PreprocResult ImagePreprocess::process(const VideoFrame& raw) {
    PreprocResult r;
    if (!raw.is_valid || raw.size == 0) return r;

    std::vector<uint8_t> rgb;
    uint32_t ow = 0, oh = 0;
    if (raw.format == PixelFormat::kMjpg) {
        if (!decode_mjpg_to_rgb(raw.data, raw.size, rgb, ow, oh)) return r;
    } else if (raw.format == PixelFormat::kYuyv) {
        ow = raw.width; oh = raw.height;
        yuyv_to_rgb(raw.data, ow, oh, rgb);
    } else {
        // NV12 或其它：略，直接复制（按 RGB 假设）
        ow = raw.width; oh = raw.height;
        rgb.assign(raw.data, raw.data + raw.size);
    }

    // 计算当前帧亮度，判断是否进入夜间模式
    uint8_t cur_brightness = compute_brightness(rgb, ow, oh);
    is_night_mode_ = (cur_brightness < cfg_.brightness_threshold);
    
    // 处理策略：
    // - 白天：做轻度对比度增强（正常画面直接识别）
    // - 夜间：不做过度增强（避免把画面拉白），交给检测器内部的 CLAHE 处理
    //   （检测器收到原始画面后自己增强，能更好地保留人脸特征）
    // 注意：这里只做降噪，对比度增强交给检测器，避免双重增强导致过曝
    if (cfg_.enable_denoise) temporal_denoise(rgb, ow, oh);

    // 1) 编码分支：resize 到 enc_width x enc_height，转 NV12
    std::vector<uint8_t> enc_rgb(static_cast<size_t>(cfg_.enc_width) * cfg_.enc_height * 3);
    resize_bilinear(rgb.data(), ow, oh, enc_rgb.data(), cfg_.enc_width, cfg_.enc_height, 3);

    std::vector<uint8_t> nv12;
    rgb_to_nv12(enc_rgb.data(), cfg_.enc_width, cfg_.enc_height, nv12);

    auto vframe = std::make_shared<VideoFrame>();
    vframe->copy_from(nv12.data(), nv12.size());
    vframe->width = cfg_.enc_width; vframe->height = cfg_.enc_height;
    vframe->format = PixelFormat::kNv12;
    vframe->pts_ms = raw.pts_ms; vframe->seq = raw.seq; vframe->is_valid = true;
    r.enc_frame = vframe;

    // 2) AI 分支：resize 到模型输入
    r.model_input_rgb.resize(static_cast<size_t>(cfg_.model_width) * cfg_.model_height * 3);
    resize_bilinear(rgb.data(), ow, oh, r.model_input_rgb.data(),
                    cfg_.model_width, cfg_.model_height, 3);
    r.model_w = cfg_.model_width;
    r.model_h = cfg_.model_height;
    r.ok = true;
    return r;
}

}  // namespace dms
