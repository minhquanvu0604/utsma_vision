#include "pcl_k_means.hpp"


KMeans::KMeans(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc):
  pc_(pc),
  low_res_pc_(std::make_shared<pcl::PointCloud<pcl::PointXYZ>>()),
  config_(kmeans::config())
{

}
KMeans::~KMeans(){}

void KMeans::load_default_config(){
  config_ = kmeans::config();
}
void KMeans::load_config(const kmeans::config& config){
  config_ = config;
}

bool KMeans::generateClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters){
  if (config_.ground_removal.enabled == true)
    removeGround();

  // Constants
  const uint32_t n = pc_->points.size(); // Num of points to process
  const uint32_t n_low_res = low_res_pc_->size();
  
  // Create random number generator to select 
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist6(0, n-1);
  
  Eigen::MatrixX3f best_centroids;
  if (config_.kmeans.search_best_k == true){
    double best_score = 0;
    for (uint32_t k = config_.kmeans.k_lower_bound; k <= config_.kmeans.k_higher_bound; k++){
      static Eigen::MatrixX3f centroids = Eigen::MatrixX3f::Zero(k, 3);

      static uint32_t rand_index;
      for (unsigned int i = 0; i < k; i++){
        rand_index = dist6(rng);
        centroids(i, 0) = pc_->points[rand_index].x;
        centroids(i, 1) = pc_->points[rand_index].y;
        centroids(i, 2) = pc_->points[rand_index].z;
      }

      calculate_centroids(low_res_pc_, centroids);
      double score = BIC_scoring(low_res_pc_, centroids);
      if (score > best_score)
        best_centroids = centroids;
    }
  }
  else {
    best_centroids = Eigen::MatrixX3f::Zero(config_.kmeans.k_lower_bound, 3);
    static uint32_t rand_index;
    for (unsigned int i = 0; i < config_.kmeans.k_lower_bound; i++){
      rand_index = dist6(rng);
      best_centroids(i, 0) = pc_->points[rand_index].x;
      best_centroids(i, 1) = pc_->points[rand_index].y;
      best_centroids(i, 2) = pc_->points[rand_index].z;
    }
    calculate_centroids(low_res_pc_, best_centroids);
  }
  

  const uint32_t best_k = best_centroids.rows();

  Eigen::MatrixXf distances(n, best_k);
  Eigen::VectorX<size_t> point_to_cluster(n);
  // Calculate Distance from each point to each centroid
    for (size_t i = 0; i < n; i++){
      for (size_t j = 0; j < best_k; j++){
        distances(i, j) = sqrt(pow(pc_->points[i].x - best_centroids(j,0), 2) + pow(pc_->points[i].y - best_centroids(j,1), 2) + pow(pc_->points[i].z - best_centroids(j,2), 2));
      }
    }
    // Find the min distance to a centroid for each point and store its index
    for (size_t row = 0; row < distances.rows(); ++row)
        distances.row(row).minCoeff(&point_to_cluster(row));

    // Create pointcloud for each cluster
    clusters.reserve(best_k);
    for (int i = 0; i < best_k; i++){
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster (new pcl::PointCloud<pcl::PointXYZ>);
      cluster->reserve(pc_->size());
      clusters.push_back(cluster);
    }

    // Add each point to respective cluster
    for (size_t i = 0; i < n; i++){
      size_t &cluster_index = point_to_cluster(i);
      clusters[cluster_index]->push_back(pc_->points[i]);
    }

  return true;
}

bool KMeans::calculate_centroids(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, Eigen::MatrixX3f& centroids){
  uint32_t k = centroids.rows();
  uint32_t n = pc->size();

  Eigen::MatrixX3f old_centroids = Eigen::MatrixX3f::Zero(k, 3);
  Eigen::MatrixXf distances(n, k);
  Eigen::VectorX<size_t> point_to_cluster(n);
  Eigen::VectorX<uint32_t> num_points_per_cluster(k);
  Eigen::MatrixX3f diff_matrix(k, 3);
  for(uint32_t iterations = 0; iterations < 10; iterations++){
    // Calculate Distance from each point to each centroid
    for (size_t i = 0; i < n; i++){
      for (size_t j = 0; j < k; j++){
        distances(i, j) = sqrt(pow(pc->points[i].x - centroids(j,0), 2) + pow(pc->points[i].y - centroids(j,1), 2) + pow(pc->points[i].z - centroids(j,2), 2));
      }
    }
    // Find the min distance to a centroid for each point and store its index
    for (size_t row = 0; row < distances.rows(); ++row)
        distances.row(row).minCoeff(&point_to_cluster(row));
    
    // Calculate new centroids
    // Cx == (p1x + p2x ... + pfx )/n
    old_centroids = centroids;
    centroids.setZero(k, 3);
    num_points_per_cluster.setZero(k);
    for (size_t i = 0; i < n; i++){
      size_t &cluster_index = point_to_cluster(i);
      num_points_per_cluster(cluster_index) += 1;
      centroids(cluster_index, 0) += pc->points[i].x;
      centroids(cluster_index, 1) += pc->points[i].y;
      centroids(cluster_index, 2) += pc->points[i].z;
    }
    // Divide sum by number of points in each cluster to get new centroids
    for (size_t i = 0; i < k; i++){
      centroids.row(i) = centroids.row(i) / num_points_per_cluster(i);
    }
    
    // Calculate difference between this and the old centroids
    diff_matrix = centroids - old_centroids;
    diff_matrix = diff_matrix.cwiseAbs();
    std::cout << "Centroids:\n" << centroids << '\n';
    std::cout << "mean_diff: " << diff_matrix.mean() << '\n';
    // If the centroids didn't move much we have converged so exit
    if (diff_matrix.mean() < 0.01)
      return true;
  }

  std::cout << "K-Means Failed to converge within the max number of iterations" << '\n';
  return false;
}

double KMeans::BIC_scoring(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, const Eigen::MatrixX3f& centroids){
  double score = 0.0;

  return score;
}

void KMeans::downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_filtered){
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud (pc);
  vg.setLeafSize (config_.downsampling.leaf_dim[0], config_.downsampling.leaf_dim[1], config_.downsampling.leaf_dim[2]);
  vg.filter (*pc_filtered);
}

bool KMeans::removeGround(){
  // Params
  uint16_t num_iterations = 2;
  double dist_threshold = 0.01;
  uint16_t seeds_precision = 100;
  uint32_t num_points_LPR = 50;
  double seed_threshold = 0.05;

  std::vector<Eigen::Vector3f> seeds;

  // Extract Initial Seeds
  {

    // Sort by lowest z value first
    pcl::PointCloud<pcl::PointXYZ> pc_sorted;
    pc_sorted.resize(num_points_LPR);
    std::partial_sort_copy(pc_->begin(), pc_->end(), pc_sorted.begin(), pc_sorted.begin() + num_points_LPR,
      [](pcl::PointXYZ a, pcl::PointXYZ b)
      {
        return a.z < b.z;
      }
    );

    // Calculate LPR
    pcl::PointXYZ LPR = {0.0f, 0.0f, 0.0f};
    for (auto it = pc_sorted.begin(); it != (pc_sorted.begin() + num_points_LPR); it++){
      LPR.x += it->x;
      LPR.y += it->y;
      LPR.z += it->z;
    }
    LPR.x = LPR.x / num_points_LPR;
    LPR.y = LPR.y / num_points_LPR;
    LPR.z = LPR.z / num_points_LPR;

    // TODO: This could be optomised by using the sorted point cloud instead so we don't have to search the whole point cloud
    // Get seeds
    for(auto it = pc_->points.begin(); it < pc_->points.end(); it+=seeds_precision){
      if (it->z < LPR.z + seed_threshold){
        Eigen::Vector3f seed;
        seed << it->x, it->y, it->z;
        seeds.push_back(seed);
      }
    }
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
bool KMeans::euclidian_distance_segmentation(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters){
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
  while (pc_lowres->size() > 0.3 * n_lowres_start)
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
