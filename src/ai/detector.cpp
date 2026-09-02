#include "ai/detector.h"

#include <algorithm>
#include <cmath>

namespace dms {

// EAR：6 点法（上 2 / 下 2 / 内外角 2）
// 公式：|p1-p5| + |p2-p4| / (2 * |p0-p3|)
float calc_ear(const Point2D p[6]) {
    // p0,p3: 水平方向（眼角）；p1,p5 上眼睑；p2,p4 下眼睑
    auto dist = [](const Point2D& a, const Point2D& b) {
        float dx = a.x - b.x, dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    float v1 = dist(p[1], p[5]);
    float v2 = dist(p[2], p[4]);
    float h  = dist(p[0], p[3]);
    if (h < 1e-3f) return 0.0f;
    return (v1 + v2) / (2.0f * h);
}

float calc_mar(const Point2D& mouth_left, const Point2D& mouth_right,
               const Point2D& mouth_top, const Point2D& mouth_bottom) {
    auto dist = [](const Point2D& a, const Point2D& b) {
        float dx = a.x - b.x, dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    };
    float v = dist(mouth_top, mouth_bottom);
    float h = dist(mouth_left, mouth_right);
    if (h < 1e-3f) return 0.0f;
    return v / h;
}

void estimate_head_pose(const std::vector<Point2D>& lm, uint32_t iw, uint32_t ih,
                        float& pitch, float& yaw, float& roll) {
    // 轻量级近似：用 2D 关键点几何关系估角度，避免完整 PnP 求解
    if (lm.size() < 5) {
        pitch = yaw = roll = 0;
        return;
    }

    // 5 点模型：[0]左眼 [1]右眼 [2]鼻 [3]左嘴 [4]右嘴
    if (lm.size() < 11) {
        // roll：两眼连线倾角
        float dx = lm[1].x - lm[0].x;
        float dy = lm[1].y - lm[0].y;
        roll = std::atan2(dy, std::max(dx, 1.0f)) * 180.0f / float(M_PI);

        // yaw：鼻尖到两眼中点的水平偏移 / 眼距
        float eye_mid_x = (lm[0].x + lm[1].x) / 2.0f;
        float eye_dist = std::max(std::abs(lm[1].x - lm[0].x), 1.0f);
        yaw = (lm[2].x - eye_mid_x) / eye_dist * 60.0f;

        // pitch：鼻尖到两眼中点的垂直距离 / 眼距
        float eye_mid_y = (lm[0].y + lm[1].y) / 2.0f;
        float vert = (lm[2].y - eye_mid_y);
        pitch = (vert / eye_dist - 0.6f) * 90.0f;
        pitch = std::clamp(pitch, -60.0f, 60.0f);
        return;
    }

    // 11 点模型：
    //   [0]左眼外 [1]左眼内 [2]右眼外 [3]右眼内 [4]鼻尖
    //   [5]嘴角左 [6]嘴角右 [7]左上睑 [8]左下睑 [9]右上睑 [10]右下睑
    // roll：两眼连线倾角
    float dx = lm[3].x - lm[0].x;
    float dy = lm[3].y - lm[0].y;
    roll = std::atan2(dy, std::max(dx, 1.0f)) * 180.0f / float(M_PI);

    // yaw：鼻尖到两眼中点的水平偏移 / 眼距
    float eye_mid_x = (lm[0].x + lm[3].x) / 2.0f;
    float eye_dist = std::max(std::abs(lm[3].x - lm[0].x), 1.0f);
    yaw = (lm[4].x - eye_mid_x) / eye_dist * 60.0f;

    // pitch：鼻尖到两眼中点的垂直距离 / 眼距
    float eye_mid_y = (lm[0].y + lm[3].y) / 2.0f;
    float vert = (lm[4].y - eye_mid_y);
    pitch = (vert / eye_dist - 0.6f) * 90.0f;
    pitch = std::clamp(pitch, -60.0f, 60.0f);
}

void map_to_image(FaceBox& box, float scale_x, float scale_y,
                  int off_x, int off_y) {
    box.x = box.x * scale_x + off_x;
    box.y = box.y * scale_y + off_y;
    box.w *= scale_x;
    box.h *= scale_y;
    for (auto& p : box.landmarks) {
        p.x = p.x * scale_x + off_x;
        p.y = p.y * scale_y + off_y;
    }
}

}  // namespace dms
