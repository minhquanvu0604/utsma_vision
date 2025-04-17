#ifndef __PCL_EUC_DIST_SEG_HPP__
#define __PCL_EUC_DIST_SEG_HPP__
#define PCL_NO_PRECOMPILE
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <random>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include <algorithm>
#include <chrono>

struct Config {
    struct downsample {
        bool enabled = true;
        std::vector<float> leaf_dim = {0.01f, 0.01f, 0.01f};
    } downsampling;

    struct euc_dist_seg {
        uint32_t max_planar_iterations = 10;
        float planar_threshold = 0.01f;

        float cluster_tolerance = 0.04f;
        uint32_t min_cluster_size = 100;
        uint32_t max_cluster_size = 25000;
    } euc_dist_seg;

    struct ground_removal {
        bool enabled = true;
        uint32_t max_iterations = 3;
        float dist_threshold = 0.01;
        uint32_t seeds_precision = 100;
        uint32_t num_points_LPR = 50;
        float seed_threshold = 0.05;
    } ground_removal;
};

class EucDistSeg
{
public:
    EucDistSeg(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
    ~EucDistSeg();

    void load_default_config();
    void load_config(const Config& config);
    
    bool removeGround();
    void downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_filtered);

    bool euclidian_distance_segmentation(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);

private:

//private:
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc_filtered_;
    Config config_;
};


#endif