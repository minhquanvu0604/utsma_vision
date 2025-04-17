#include "euc_dist_seg.hpp"


EucDistSeg::EucDistSeg(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc):
  pc_(pc),
  config_(Config()),
  pc_filtered_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>())
{
  //pc_filtered_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  pc_filtered_->reserve(pc_->size());
  for (int i = 0; i < pc_->size(); i++){
    pcl::PointXYZ point = pc_->at(i);
    if (isnan(point.x) || isnan(point.y) || isnan(point.z))
      continue;
    pc_filtered_->push_back(point);
  }
}
EucDistSeg::~EucDistSeg(){}

void EucDistSeg::load_default_config(){
  config_ = Config();
}
void EucDistSeg::load_config(const Config& config){
  config_ = config;
}

void EucDistSeg::downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_filtered){
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud (pc);
  vg.setLeafSize (config_.downsampling.leaf_dim[0], config_.downsampling.leaf_dim[1], config_.downsampling.leaf_dim[2]);
  vg.filter (*pc_filtered);
}

bool EucDistSeg::removeGround(){
  // Params
  uint16_t num_iterations = 10;
  double dist_threshold = 0.01;
  uint16_t seeds_precision = 10;
  uint32_t num_points_LPR = 50;
  double seed_threshold = 0.05;

  std::vector<Eigen::Vector3f> seeds;

  if (pc_filtered_->size() < num_points_LPR)
    std::cout << "Not enough points to create LPR: " << pc_filtered_->size() << std::endl;
  // Extract Initial Seeds
  {
    // Sort by lowest z value first
    pcl::PointCloud<pcl::PointXYZ> pc_sorted;
    pc_sorted.resize(num_points_LPR);
    std::partial_sort_copy(pc_filtered_->begin(), pc_filtered_->end(), pc_sorted.begin(), pc_sorted.begin() + num_points_LPR,
      [](pcl::PointXYZ a, pcl::PointXYZ b)
      {
        return a.z < b.z;
      }
    );
    std::cout << "pcsortedsize: " << pc_sorted.size() << std::endl;

    // Calculate LPR
    pcl::PointXYZ LPR = {0.0f, 0.0f, 0.0f};
    for (auto it = pc_sorted.begin(); it != (pc_sorted.begin() + num_points_LPR); it++){
      LPR.x += it->x;
      LPR.y += it->y;
      LPR.z += it->z;
    }
    std::cout << "LPR: " << LPR.x << ", " << LPR.y << ", " << LPR.z << std::endl;
    LPR.x = LPR.x / num_points_LPR;
    LPR.y = LPR.y / num_points_LPR;
    LPR.z = LPR.z / num_points_LPR;
    std::cout << "LPR: " << LPR.x << ", " << LPR.y << ", " << LPR.z << std::endl;

    // TODO: This could be optomised by using the sorted point cloud instead so we don't have to search the whole point cloud
    // Get seeds
    for(auto it = pc_->points.begin(); it < pc_->points.end(); it+=seeds_precision){
      if (it->z < LPR.z + seed_threshold){
        Eigen::Vector3f seed;
        seed << it->x, it->y, it->z;
        seeds.push_back(seed);
      }
    }
    std::cout << "Num seeds: " << seeds.size() << std::endl;
  }

  // Ground Plane Fitting
  Eigen::Vector3f normal;
  float a,b,c,d;
  float point_dist_from_plane;
  for (uint16_t i = 0; i < num_iterations; i++){
    // Estimate Plane ax + by + cz + d = 0
    {
      // Calculate geometric mean point of all seeds s_hat
      Eigen::Vector3f s_hat = Eigen::Vector3f::Zero(3, 1);
      for (auto seed_it = seeds.begin(); seed_it < seeds.end(); seed_it++){
        s_hat = s_hat + *seed_it;
      }
      s_hat = s_hat / seeds.size();

      // Calculate covariance matrix
      Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
      Eigen::Vector3f point_diff;
      for (auto seed_it = seeds.begin(); seed_it < seeds.end(); seed_it++){
        point_diff = *seed_it - s_hat;
        covariance = covariance + (point_diff * point_diff.transpose());
      }

      // Singular Value Decomposition of Covariance matrix
      Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::ComputeFullU> svd(covariance, Eigen::ComputeFullU);
      
      // Find smallest singular value index
      Eigen::Vector3f singular_values = svd.singularValues().real();
      uint8_t normal_col;
      if (singular_values(0) < singular_values(1))
      {
        if (singular_values(0) < singular_values(2))
          normal_col = 0;
        else
          normal_col = 2;
      }
      else
      {
        if (singular_values(1) < singular_values(2))
          normal_col = 1;
        else
          normal_col = 2;
      }

      // Get normal of plane
      normal = svd.matrixU().col(normal_col);

      // Calculate coeficents of plane equation
      a = normal(0);
      b = normal(1);
      c = normal(2);
      d = normal.transpose() * s_hat;
      d = -d;

    }
    std::cout << "abcd: " << a << ", " << b << ", " << c << ", " << d << std::endl;
    // If this is not the last iteration
    if (i != num_iterations - 1){
      seeds.clear();
      // Calculate points that are part of the current plane
      for (auto it = pc_->points.begin(); it != pc_->points.end(); it+=seeds_precision){
        point_dist_from_plane = a * it->x + b * it->y + c * it->z + d;
        if (point_dist_from_plane < dist_threshold){
          Eigen::Vector3f seed;
          seed << it->x, it->y, it->z;
          seeds.push_back(seed);
        }
      }
    }
    // If this is the last iteration we have the final plane so remove ground points from the point cloud instead of saving the ground points
    else{
      std::cout << "final abcd: " << a << ", " << b << ", " << c << ", " << d << std::endl;
      // Create new pointcloud to store the filtered points into
      pcl::PointCloud<pcl::PointXYZ> no_ground_pc;
      no_ground_pc.reserve(pc_->size()  - (0.5 * seeds.size() * seeds_precision));

      // Iterate through all points, if any point is close enough to the plane ignore it
      for (auto it = pc_->points.begin(); it != pc_->points.end(); it++){
        point_dist_from_plane = a * it->x + b * it->y + c * it->z + d;
        if (point_dist_from_plane >= dist_threshold){
          no_ground_pc.points.push_back(*it);
        }
      }
      *pc_ = no_ground_pc;
    }
  }
  pc_->resize(pc_->points.size());
  return true;
}

bool EucDistSeg::euclidian_distance_segmentation(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters){
  auto &config = config_.euc_dist_seg; 
  auto start_time = std::chrono::system_clock::now();

  start_time = std::chrono::system_clock::now();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc_lowres (new pcl::PointCloud<pcl::PointXYZ>);
  downsample(pc_, pc_lowres);
  std::cout << "Downsampling time (ms): " <<  (std::chrono::system_clock::now() - start_time).count() / 1e6 << '\n';
  std::cout << "PointCloud before filtering has: " << pc_->size () << " data points." << std::endl; 
  std::cout << "PointCloud after filtering has: " << pc_lowres->size ()  << " data points." << std::endl; 

  // Create the segmentation object for the planar model and set all the parameters
  start_time = std::chrono::system_clock::now();
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(config.max_planar_iterations);
  seg.setDistanceThreshold(config.planar_threshold);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  uint32_t n_lowres_start = pc_lowres->size();
  while (pc_lowres->size() > 0.5 * n_lowres_start)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(pc_lowres);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Remove the planar inliers, extract the rest
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(pc_lowres);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*cloud_f);
    *pc_lowres = *cloud_f;
  }
  std::cout << "Planar extraction time (ms): " <<  (std::chrono::system_clock::now() - start_time).count() / 1e6 << '\n';

  start_time = std::chrono::system_clock::now();
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(pc_lowres);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(config.cluster_tolerance);
  ec.setMinClusterSize (config.min_cluster_size);
  ec.setMaxClusterSize (config.max_cluster_size);
  ec.setSearchMethod (tree);
  ec.setInputCloud (pc_lowres);
  ec.extract (cluster_indices);

  clusters.clear();
  clusters.reserve(cluster_indices.size());
  
  int j = 0;
  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*pc_lowres)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
    clusters.push_back(cloud_cluster);
    j++;
  }
  std::cout << "Euclidean Cluster Extraction Time (ms): " <<  (std::chrono::system_clock::now() - start_time).count() / 1e6 << '\n';

  return true;
}
