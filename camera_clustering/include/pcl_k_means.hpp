#ifndef __PCL_K_MEANS_HPP__
#define __PCL_K_MEANS_HPP__
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
#include "unsupported/Eigen/CXX11/Tensor"
#include <algorithm>
#include <chrono>

namespace kmeans{

struct config {
    struct downsample {
        bool enabled = true;
        std::vector<float> leaf_dim = {0.01f, 0.01f, 0.01f};
    } downsampling;

    struct kmeans {
        bool search_best_k = true;
        uint32_t k_lower_bound = 1;
        uint32_t k_higher_bound = 10;

        uint32_t max_iterations = 5;
        float convergence_threshold = 0.01;

        bool output_lowres_pointcloud = false;
    } kmeans;

    struct euc_dist_seg {
        uint32_t max_planar_iterations = 10;
        float planar_threshold = 0.02f;

        float cluster_tolerance = 0.02f;
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
}

class KMeans
{
public:
    KMeans(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc);
    ~KMeans();

    void load_default_config();
    void load_config(const kmeans::config& config);

    bool generateClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);
    bool calculate_centroids(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::MatrixX3f& centroids);
    double BIC_scoring(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, const Eigen::MatrixX3f& centroids);
    
    bool removeGround();
    void downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_filtered);

    bool euclidian_distance_segmentation(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters);

private:

//private:
public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr low_res_pc_;
    kmeans::config config_;
};


#endif