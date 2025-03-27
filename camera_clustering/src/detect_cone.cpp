#include "detect_cone.hpp"

ConeDetector::ConeDetector(cone_detector::config config):
    config_(config)
{

}


std::vector<pcl::PointXYZ> ConeDetector::detect_cones(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters){
    std::vector<pcl::PointXYZ> cones;
    
    return cones;
}