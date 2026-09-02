#include "ai/mock_detector.h"

#include <cmath>

#include "utils/log.h"

namespace dms {

bool MockDetector::detect(const uint8_t* /*rgb*/, uint32_t w, uint32_t h,
                          DetectionResult& result) {
    ++frame_idx_;

    FaceBox face;
    face.score = 0.98f;
    // 画面中心 40% 区域
    face.w = w * 0.4f;
    face.h = h * 0.5f;
    face.x = (w - face.w) / 2.0f;
    face.y = (h - face.h) / 2.0f;

    // 11 个关键点（与 detector.h estimate_head_pose 约定一致）
    float cx = w / 2.0f, cy = h / 2.0f;
    float ex = w * 0.10f;   // 眼睛水平半距
    float ey_o = h * 0.06f; // 眼睛垂直半距
    // 模拟眨眼：每 3 秒（~30fps推理 → 90帧）一次，闭眼持续 ~5 帧
    bool blinking = ((frame_idx_ % 90) < 5);
    // 长时间闭眼（疲劳 / 模拟）
    bool long_closed = simulated_fatigue_ || ((frame_idx_ / 300) % 4 == 3);  // 每 4 个 10s 周期里第 4 个 10s 闭眼

    float ear_top_y = cy - h * 0.05f - (blinking || long_closed ? ey_o * 0.8f : 0);
    float ear_bot_y = cy - h * 0.05f + (blinking || long_closed ? ey_o * 0.8f : ey_o);

    face.landmarks.resize(11);
    face.landmarks[0]  = {cx - ex, cy - h * 0.05f}; // 左眼外
    face.landmarks[1]  = {cx - ex * 0.4f, cy - h * 0.05f}; // 左眼内
    face.landmarks[2]  = {cx + ex, cy - h * 0.05f}; // 右眼外
    face.landmarks[3]  = {cx + ex * 0.4f, cy - h * 0.05f}; // 右眼内
    face.landmarks[4]  = {cx, cy + h * 0.02f};      // 鼻尖
    face.landmarks[5]  = {cx - ex * 0.6f, cy + h * 0.12f}; // 嘴左
    face.landmarks[6]  = {cx + ex * 0.6f, cy + h * 0.12f}; // 嘴右
    face.landmarks[7]  = {cx - ex * 0.7f, ear_top_y}; // 左上睑
    face.landmarks[8]  = {cx - ex * 0.7f, ear_bot_y}; // 左下睑
    face.landmarks[9]  = {cx + ex * 0.7f, ear_top_y}; // 右上睑
    face.landmarks[10] = {cx + ex * 0.7f, ear_bot_y}; // 右下睑

    // 计算 EAR（左眼：landmark 0,1,7,8,1,0 简化为 6 点）
    Point2D left_eye[6] = {
        face.landmarks[0], face.landmarks[7], face.landmarks[1],
        face.landmarks[1], face.landmarks[8], face.landmarks[0]
    };
    Point2D right_eye[6] = {
        face.landmarks[3], face.landmarks[9], face.landmarks[2],
        face.landmarks[2], face.landmarks[10], face.landmarks[3]
    };
    face.ear_left  = calc_ear(left_eye);
    face.ear_right = calc_ear(right_eye);

    // MAR（轻微张开）
    Point2D mouth_top = {cx, cy + h * 0.08f};
    Point2D mouth_bot = {cx, cy + h * 0.16f};
    face.mar = calc_mar(face.landmarks[5], face.landmarks[6], mouth_top, mouth_bot);

    // 头部姿态
    estimate_head_pose(face.landmarks, w, h,
                       face.head_pitch, face.head_yaw, face.head_roll);

    result.faces.push_back(std::move(face));
    if (frame_idx_ == 1) {
        LOGI("MOCK", "MockDetector running (w=%u h=%u), EAR-L=%.2f R=%.2f",
             w, h, result.faces[0].ear_left, result.faces[0].ear_right);
    }
    return true;
}

}  // namespace dms
