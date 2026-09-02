#include "ai/rknn_detector.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#ifdef DMS_HAS_RKNN
#include "rknn_api.h"
#endif

#include "utils/log.h"

namespace dms {

RknnDetector::RknnDetector(std::string model_path)
    : model_path_(std::move(model_path)) {}

RknnDetector::~RknnDetector() {
#ifdef DMS_HAS_RKNN
    if (rknn_ctx_) rknn_destroy(*static_cast<rknn_context*>(rknn_ctx_));
#endif
}

#ifdef DMS_HAS_RKNN

// ==================== 真实 RKNN 实现 ====================

bool RknnDetector::init() {
    // 1) 读取 .rknn 模型文件
    std::ifstream ifs(model_path_, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        LOGE("RKNN", "open model %s failed", model_path_.c_str());
        return false;
    }
    size_t sz = ifs.tellg();
    std::vector<uint8_t> model(sz);
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(model.data()), sz);

    // 2) rknn_init
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model.data(), model.size(), 0, nullptr);
    if (ret < 0) { LOGE("RKNN", "rknn_init failed: %d", ret); return false; }
    rknn_ctx_ = new rknn_context(ctx);

    // 3) 查询输入/输出属性（拿 input shape）
    rknn_input_output_num io_num{};
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (io_num.n_input < 1) { LOGE("RKNN", "no input tensor"); return false; }

    rknn_tensor_attr in_attr{};
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    // NHWC：in_attr.fmt == RKNN_TENSOR_NHWC，dims = [N,H,W,C]
    in_h_ = in_attr.dims[1];
    in_w_ = in_attr.dims[2];
    LOGI("RKNN", "model: %s input=%ux%u output_num=%d",
         model_path_.c_str(), in_w_, in_h_, io_num.n_output);

    // 绑核（NPU 三核）
    if (cfg_.core_mask != 0) {
        rknn_core_mask core =
            static_cast<rknn_core_mask>(cfg_.core_mask);
        rknn_set_core_mask(ctx, core);
    }

    out_bboxes_.resize(cfg_.num_anchors * 5);
    out_kps_.resize(cfg_.num_anchors * 2 * cfg_.num_landmarks);
    initialized_ = true;
    return true;
}

bool RknnDetector::run_inference(const uint8_t* rgb) {
    rknn_context ctx = *static_cast<rknn_context*>(rknn_ctx_);

    // 设置输入：模型输入一般 NHWC uint8（INT8 量化），归一化在模型内做
    rknn_input input{};
    input.index   = 0;
    input.type    = RKNN_TENSOR_UINT8;
    input.fmt     = RKNN_TENSOR_NHWC;
    input.size    = in_w_ * in_h_ * 3;
    input.buf     = const_cast<uint8_t*>(rgb);
    input.pass_through = 1;  // 不再做内部转换
    rknn_inputs_set(ctx, 1, &input);

    int ret = rknn_run(ctx, nullptr);
    if (ret < 0) { LOGE("RKNN", "rknn_run failed: %d", ret); return false; }

    // 取输出（0: boxes, 1: landmarks）
    rknn_output outputs[2] = {};
    outputs[0].want_float = 1;
    outputs[1].want_float = 1;
    outputs[0].is_prealloc = 1;
    outputs[1].is_prealloc = 1;
    outputs[0].buf = out_bboxes_.data();
    outputs[0].size = out_bboxes_.size() * sizeof(float);
    outputs[1].buf = out_kps_.data();
    outputs[1].size = out_kps_.size() * sizeof(float);
    rknn_outputs_get(ctx, 2, outputs, nullptr);

    return true;
}

void RknnDetector::postprocess(DetectionResult& out) {
    struct Cand { FaceBox box; std::vector<Point2D> kps; float score; };
    std::vector<Cand> cands;
    cands.reserve(256);

    for (int i = 0; i < cfg_.num_anchors; ++i) {
        const float* b = &out_bboxes_[i * 5];
        float score = b[4];
        if (score < cfg_.score_thresh) continue;

        Cand c;
        // b[0..1] 为中心点坐标，转换为左上角
        c.box.x = b[0] - b[2] / 2;
        c.box.y = b[1] - b[3] / 2;
        c.box.w = b[2];
        c.box.h = b[3];
        c.box.score = score;
        // 关键点
        c.kps.resize(cfg_.num_landmarks);
        for (int k = 0; k < cfg_.num_landmarks; ++k) {
            c.kps[k].x = out_kps_[i * 2 * cfg_.num_landmarks + 2 * k];
            c.kps[k].y = out_kps_[i * 2 * cfg_.num_landmarks + 2 * k + 1];
        }
        c.score = score;
        cands.push_back(std::move(c));
    }

    // NMS
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });
    std::vector<uint8_t> suppressed(cands.size(), 0);
    auto iou = [](const FaceBox& a, const FaceBox& b) {
        float xx1 = std::max(a.x, b.x), yy1 = std::max(a.y, b.y);
        float xx2 = std::min(a.x + a.w, b.x + b.w), yy2 = std::min(a.y + a.h, b.y + b.h);
        float w = std::max(0.0f, xx2 - xx1), h = std::max(0.0f, yy2 - yy1);
        float inter = w * h;
        float uni = a.w * a.h + b.w * b.h - inter;
        return uni > 0 ? inter / uni : 0;
    };
    for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (suppressed[j]) continue;
            if (iou(cands[i].box, cands[j].box) > cfg_.nms_thresh)
                suppressed[j] = 1;
        }
    }

    // 取保留的（最多前 5 个）
    int kept = 0;
    for (size_t i = 0; i < cands.size() && kept < 5; ++i) {
        if (suppressed[i]) continue;
        FaceBox box = cands[i].box;
        box.landmarks = cands[i].kps;

        // 由关键点计算 EAR/MAR/姿态
        // 约定 5 点：[0]左眼 [1]右眼 [2]鼻 [3]左嘴 [4]右嘴
        if (box.landmarks.size() >= 5) {
            // 简化 EAR：用眼角 + 鼻/嘴角垂直距离近似
            float eye_dist = std::abs(box.landmarks[1].x - box.landmarks[0].x);
            float eye_nose = std::abs(box.landmarks[2].y -
                                     (box.landmarks[0].y + box.landmarks[1].y) / 2);
            box.ear_left  = std::clamp(eye_nose / std::max(eye_dist, 1.0f), 0.05f, 0.5f);
            box.ear_right = box.ear_left;

            float mouth_dist = std::abs(box.landmarks[4].x - box.landmarks[3].x);
            float nose_mouth = std::abs(box.landmarks[2].y -
                                       (box.landmarks[3].y + box.landmarks[4].y) / 2);
            box.mar = nose_mouth / std::max(mouth_dist, 1.0f);

            estimate_head_pose(box.landmarks, in_w_, in_h_,
                               box.head_pitch, box.head_yaw, box.head_roll);
        }
        out.faces.push_back(std::move(box));
        ++kept;
    }
}

bool RknnDetector::detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                          DetectionResult& result) {
    if (!initialized_) { LOGE("RKNN", "not inited"); return false; }
    if (w != in_w_ || h != in_h_) {
        LOGW("RKNN", "input %ux%u != model %ux%u", w, h, in_w_, in_h_);
        return false;
    }
    uint64_t t0 = now_ms();
    if (!run_inference(rgb)) return false;
    postprocess(result);
    result.inference_ms = static_cast<int>(now_ms() - t0);
    return true;
}

#else  // !DMS_HAS_RKNN

// ==================== Stub：实现同一接口，转发到 Mock 行为 ====================
bool RknnDetector::init() {
    LOGW("RKNN", "[STUB] RKNN lib not linked, fallback to no-op detector "
         "(model=%s)", model_path_.c_str());
    in_w_ = 320; in_h_ = 240;
    initialized_ = true;
    return true;
}

bool RknnDetector::run_inference(const uint8_t*) { return true; }

void RknnDetector::postprocess(DetectionResult& /*out*/) {}

bool RknnDetector::detect(const uint8_t* /*rgb*/, uint32_t /*w*/, uint32_t /*h*/,
                          DetectionResult& result) {
    // 返回空结果（无人脸）；上层 PERCLOS 容错处理
    result.inference_ms = 0;
    return true;
}

#endif  // DMS_HAS_RKNN

}  // namespace dms
