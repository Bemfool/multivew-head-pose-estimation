#ifndef MHPE_2D_RENDER_UTILS_H
#define MHPE_2D_RENDER_UTILS_H

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <vector>


namespace utils
{

inline std::vector<dlib::image_window::overlay_circle> renderPts(
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

}


#endif // MHPE_2D_RENDER_UTILS_H