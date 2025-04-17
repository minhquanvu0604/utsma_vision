#include "pcl_k_means.hpp"
#include "euc_dist_seg.hpp"
#include "gtest/gtest.h"
#include <chrono>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

using std::vector;
using pcl::PointCloud;
using pcl::PointXYZ;

TEST(table_scene, euclidian_distance_segmentation){
  PointCloud<PointXYZ>::Ptr table_scene_pc (new PointCloud<PointXYZ>);
  pcl::PCDReader reader;
  reader.read("../../test/table_scene_lms400.pcd", *table_scene_pc);

  EucDistSeg algo1(table_scene_pc);
  Config config1;
  config1.downsampling.leaf_dim = {0.01f, 0.01f, 0.01f};
  algo1.load_config(config1);

  std::cout << "Num of Points before ground removal: " << table_scene_pc->size() << '\n';
  auto start_time = std::chrono::system_clock::now();
  algo1.removeGround();
  double processing_time1 = (std::chrono::system_clock::now() - start_time).count() / 1e6;
  std::cout << "Num of Points after ground removal:  " << table_scene_pc->size() << '\n';

  vector<PointCloud<PointXYZ>::Ptr> clusters;
  start_time = std::chrono::system_clock::now();
  algo1.euclidian_distance_segmentation(clusters);
  double processing_time2 = (std::chrono::system_clock::now() - start_time).count() / 1e6;
  

  std::cout << "Ground Removal Time (ms): " << processing_time1 << '\n';
  std::cout << "Clustering Time low res (ms): " << (processing_time2) << '\n';
  std::cout << "Total Time (ms): " << (processing_time1 + processing_time2) << '\n';
  /* ASSERT_TRUE(success); */
  pcl::io::savePCDFileASCII ("../../output/table_scene_no_ground.pcd", *table_scene_pc);

  char filename_buf[100];
  for (size_t i = 0; i < clusters.size(); i++){
    auto &cluster = clusters[i];
    sprintf(filename_buf, "../../output/euc_lowres/cluster%lu.pcd", i);
    pcl::io::savePCDFileASCII (filename_buf, *cluster);
  }   
}

TEST(zed_pc, euclidian_distance_segmentation){
  char filename_buf[100];
  for (size_t i = 1; i < 5; i++){
    PointCloud<PointXYZ>::Ptr zed_pc (new PointCloud<PointXYZ>);
    pcl::PCDReader reader;
    sprintf(filename_buf, "../../test/zed_pc%lu.pcd", i);
    reader.read(filename_buf, *zed_pc);

    EucDistSeg algo1(zed_pc);
    Config config1;
    config1.downsampling.leaf_dim = {0.01f, 0.01f, 0.01f};
    algo1.load_config(config1);

    std::cout << "Num of Points before ground removal: " << zed_pc->size() << '\n';
    auto start_time = std::chrono::system_clock::now();
    //algo1.removeGround();
    double processing_time1 = (std::chrono::system_clock::now() - start_time).count() / 1e6;
    std::cout << "Num of Points after ground removal:  " << zed_pc->size() << '\n';

    vector<PointCloud<PointXYZ>::Ptr> clusters;
    start_time = std::chrono::system_clock::now();
    algo1.euclidian_distance_segmentation(clusters);
    double processing_time2 = (std::chrono::system_clock::now() - start_time).count() / 1e6;
    

    std::cout << "Ground Removal Time (ms): " << processing_time1 << '\n';
    std::cout << "Clustering Time low res (ms): " << (processing_time2) << '\n';
    std::cout << "Total Time (ms): " << (processing_time1 + processing_time2) << '\n';
    sprintf(filename_buf, "../../output/pc%lu_noground.pcd", i);
    pcl::io::savePCDFileASCII (filename_buf, *zed_pc);

    for (size_t j = 0; j < clusters.size(); j++){
      auto &cluster = clusters[j];
      sprintf(filename_buf, "../../output/zed_clustering/pc%lu_cluster%lu.pcd", i, j);
      pcl::io::savePCDFileASCII (filename_buf, *cluster);
    }   
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
