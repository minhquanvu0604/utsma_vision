#ifndef DETECT_CONE
#define DETECT_CONE

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

namespace cone_detector {
    struct config {
        int tmp = 0;
    };
}

class ConeDetector {
public:
    ConeDetector(): ConeDetector(cone_detector::config()) {};
    ConeDetector(cone_detector::config config);

    std::vector<pcl::PointXYZ> detect_cones(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);

private:

private:

    cone_detector::config config_;

};

#endif