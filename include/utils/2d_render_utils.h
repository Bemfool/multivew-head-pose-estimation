#ifndef MHPE_2D_RENDER_UTILS_H
#define MHPE_2D_RENDER_UTILS_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <vector>
#include "db_params.h"

namespace utils
{

inline std::vector<dlib::image_window::overlay_circle> RenderPts(
    const std::vector<dlib::point>&pts,
    double radius = 1.0,
    const dlib::rgb_pixel color = dlib::rgb_pixel(0,255,0),
    const std::vector<bool> validList = std::vector<bool>()
)
{
    std::vector<dlib::image_window::overlay_circle> circles;
    for(const auto& p : pts)
        circles.emplace_back(p, radius, color);
    return circles;
}

template<typename _Tp, typename _Ep>
inline void RotateToFront(_Tp& u, _Tp& v, const _Ep& height, const _Ep& width, const RotateType& type)
{
    _Tp oldU = u, oldV = v;

    if(type == RotateType_CW)
    {
        u = static_cast<_Tp>(height) - oldV;
        v = oldU;
    }
    else
    {
        u = oldV;
        v = static_cast<_Tp>(width) - oldU;
    }
}

template<typename _Tp, typename _Ep>
inline void RotateFromFront(_Tp& u, _Tp& v, const _Ep& height, const _Ep& width, RotateType type)
{
    _Tp oldU = u, oldV = v;
    if(type == RotateType_CCW)
    {
        u = static_cast<_Tp>(width) - oldV;
        v = oldU;
    }
    else
    {
        u = oldV;
        v = static_cast<_Tp>(height) - oldU;
    }
}


}


#endif // MHPE_2D_RENDER_UTILS_H