#ifndef DMS_DETECTOR_H
#define DMS_DETECTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "utils/media_types.h"

namespace dms {

// 抽象检测器接口（策略模式：Mock / RKNN 可插拔）
//
// 设计动机：
// - RKNN 推理代码只能在板子上编译运行；开发机用 Mock 跑通整条流水线
// - 上层（FatigueAnalyzer）只依赖此抽象，不关心具体实现
// - 输入：预处理后的 RGB（resize 到模型尺寸）
// - 输出：DetectionResult（已含 EAR/MAR/姿态等后处理结果）
class Detector {
public:
    virtual ~Detector() = default;

    // 加载模型 / 初始化
    virtual bool init() = 0;

    // 输入模型尺寸（rgb 输入应为 w*h*3）
    virtual void input_shape(uint32_t& w, uint32_t& h) const = 0;

    // 推理：rgb 为 RGB888 数据，w/h 为输入尺寸；result.pts_ms 由调用方设置
    // 返回是否成功（无人脸时返回 true 且 faces 为空）
    virtual bool detect(const uint8_t* rgb, uint32_t w, uint32_t h,
                        DetectionResult& result) = 0;

    virtual const char* name() const = 0;
};

// 关键点索引定义（与主流轻量级人脸关键点模型一致，5 点）
enum class LandmarkIdx : uint8_t {
    kLeftEye0 = 0, kLeftEye1, kRightEye0, kRightEye1, kNose,
    kMouthLeft, kMouthRight, kLeftEyeTop, kLeftEyeBot, kRightEyeTop, kRightEyeBot
};

// ============== 共用后处理：几何指标计算（Mock 和 RKNN 都用） ==============

// Eye Aspect Ratio：由 6 个关键点（左/右上角、左/右下角、左/右内角）计算
// landmarks 为该眼睛的 6 个点
float calc_ear(const Point2D p[6]);

// 嘴部纵横比 MAR
float calc_mar(const Point2D& mouth_left, const Point2D& mouth_right,
               const Point2D& mouth_top, const Point2D& mouth_bottom);

// 由两眼与鼻/嘴角关系估算头部姿态（pitch/yaw/roll，度）
void estimate_head_pose(const std::vector<Point2D>& lm, uint32_t iw, uint32_t ih,
                        float& pitch, float& yaw, float& roll);

// 把模型坐标系下的 bbox/关键点（0~1 归一化 或 模型像素）映射回原图坐标
void map_to_image(FaceBox& box, float scale_x, float scale_y,
                  int off_x = 0, int off_y = 0);

}  // namespace dms

#endif  // DMS_DETECTOR_H
