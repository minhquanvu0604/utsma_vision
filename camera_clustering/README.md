# UTSMA Camera Clustering Library

## Requirements
- CMake
- pcl (Point Cloud Library) v1.12+
- Eigen3

## Libraries
Has two seperate libraries implementing two different clustering algorithms:
- K-Means | Currently broken
- Euclidean Distance Segmentation | The one we will use

## Ground Removal

The purpose of this step is to find and remove the ground from the input pointcloud. Considering a large proportion of the pointcloud will be ground on the track this step will massively reduce the number of points to consider in following steps which results in a very significant performance boost.

This Algorithm is based off of a paper by Dimitris Zermas, Izzat Izzat and Nikolaos Papanikolopoulos:

[Fast Segmentation of 3D Point Clouds: A paradigm on LiDAR Data for Autonomous Vehicle Applications](https://www.researchgate.net/publication/318325507_Fast_Segmentation_of_3D_Point_Clouds_A_Paradigm_on_LiDAR_Data_for_Autonomous_Vehicle_Applications)

### Steps

1. The Ground Removal algorithm will start by taking some points with the lowest z-value, these are the "ground points".
2. Estimate the ground plane from the ground points using $ax + by + cz + d = 0$.
3. Find all points within some distance to the ground plane, these are our new ground points.
4. Repeat 2-3 some number of times.
5. Remove the ground points from the pointcloud data

### Potential Problems or Improvements
- Currently the algorithm assumes there is a ground plane, so in the case that there is not something will likely go wrong. Adding some way to handle this case would be ideal.
- Currently the algorithm iterates a set number of times, adding a way to check if the result has converged upon an answer could improve performance.

## Voxel Grid Downsampling

The purpose of this step is to reduce the resolution of the pointcloud data so that the next steps have much less data to work through.

The implementation of this is very simple as there is a class built into the pcl library that will do this for us.

### !!! Important !!!

The pcl VoxelGrid Algorithm will run extremely slowly unless compiler optimizations are turned on. Make sure when you are building with CMake that the build variant is set to "Release" and NOT "Debug". VS Code Will default to debug.

This change alone made this step go from taking ~600ms to ~30ms, so a 20x improvement.

### Steps

1. Create a grid of cubes over the pointcloud data with a specified grid size.
2. Find the centroid of all the points within each cube.
3. These centroids are our new pointcloud.

### Potential Problems or Improvements
- This algorithm will lower the range at which we can detect cones as a blanket reduced resolution will disporportionally affect far objects that are far away from the camera. This could be fixed by only applying downsampling to areas of the image with a high density of points.

## Planar Extraction

This step takes the downsampled pointcloud and uses inbuilt pcl functions to attempt to remove flat planes from the point cloud, starting from the planes with the largest number of points.

This would remove the ground, however my ground removal algorithm is way faster than this planar extraction even when the planar extraction uses a downsampled pointcloud, as such my ground removal algorithm is still used in order to increase performance.

This step is the slowest taking more than 2-3x longer than the other steps so if performance improvements are needed you should start with this step.

The implementation is based off of this tutorial on the pcl docs page: 
[Euclidean Cluster Extraction](https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html)

## Euclidean Distance Segmentation

This is the part of the algorithm that actually does the clustering. As it says in the title it clusters based off of euclidean distance.

The algorithm uses a kdtree search to find groups of points that are close to each other.

You may set a minimum and maximum amount of points for each cluster as well as the maximum distance for two points to be treated as part of the same cluster.

The implementation is based off of this tutorial on the pcl docs page: 
[Euclidean Cluster Extraction](https://pcl.readthedocs.io/projects/tutorials/en/master/cluster_extraction.html)