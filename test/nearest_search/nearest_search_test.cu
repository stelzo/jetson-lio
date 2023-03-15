#include <gtest/gtest.h>
#include <ikd-Tree/ikd_Tree.h>

#include <cuda-utils.h>
#include <cuda_runtime.h>
#include <ikd-Tree/ikd_Tree_gpu_search.h>

#include <common_lib.h>
#include <math/common.h>
#include <math/math.h>

#include <iostream>

#include "common_lib.h"

PointType createPoint(float x, float y, float z) {
    PointType p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

float calc_dist(PointXYZINormal_CUDA a, PointXYZINormal_CUDA b) {
    float dist = 0.0f;
    dist = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return dist;
}

// Important: Initialize globally to prevent Segfault
KD_TREE<PointType> ikdTree;

TEST(ikdTree, nearestSearch) {
    constexpr size_t k = 6;

    // Define pointcloud
    PointVector h_points = {
        // Points outside of range
        createPoint(2, 2, 0),
        createPoint(2, -2, 0),
        createPoint(-2, 2, 0),
        createPoint(-2, -2, 0),
        createPoint(0, 2, 2),
        createPoint(0, 2, -2),
        createPoint(0, -2, 2),
        createPoint(0, -2, -2),
        createPoint(2, 0, 2),
        createPoint(2, 0, -2),
        createPoint(-2, 0, 2),
        createPoint(-2, 0, -2),
        // Points inside of range
        // Use odd numbers to compare the points better via sorting
        createPoint(1.1, 0, 0),
        createPoint(-1.2, 0, 0),
        createPoint(0, 1.3, 0),
        createPoint(0, -1.4, 0),
        createPoint(0, 0, 1.5),
        createPoint(0, 0, -1.6),
    };

    // Initialize GPU memory
    PointType* managed_points;
    CHECK_CUDA_ERROR(cudaMallocManaged(&managed_points, h_points.size() * sizeof(PointType)));
    for (size_t i = 0; i < h_points.size(); i++) {
        managed_points[i] = PointType();
        managed_points[i].x = h_points[i].x;
        managed_points[i].y = h_points[i].y;
        managed_points[i].z = h_points[i].z;
    }

    PointVector d_points;
    d_points.insert(d_points.end(), &managed_points[0], &managed_points[h_points.size()]);

    ikdTree.Build(d_points);
    ikdTree.consistent();

    PointXYZINormal_CUDA query;
    query.x = 0;
    query.y = 0;
    query.z = 0;

    PointXYZINormal_CUDA* d_query;
    CHECK_CUDA_ERROR(cudaMallocManaged(&d_query, sizeof(PointXYZINormal_CUDA)));
    d_query[0] = query;

    PointXYZINormal_CUDA* Nearest_Points;
    CHECK_CUDA_ERROR(cudaMallocManaged(&Nearest_Points, k * sizeof(PointXYZINormal_CUDA)));
    for (size_t i = 0; i < k; i++) {
        Nearest_Points[i] = PointXYZINormal_CUDA();
    }

    int* nearest_size;
    CHECK_CUDA_ERROR(cudaMallocManaged(&nearest_size, sizeof(int)));
    nearest_size[0] = 0;

    float* pointSearchSqDis;
    CHECK_CUDA_ERROR(cudaMallocManaged(&pointSearchSqDis, k * sizeof(float)));

    size_t* dist_size;
    CHECK_CUDA_ERROR(cudaMallocManaged(&dist_size, sizeof(size_t)));
    dist_size[0] = k;

    // Perform search
    Raw_Nearest_Search((void*)ikdTree.Root_Node, d_query, k, Nearest_Points, nearest_size,
                       pointSearchSqDis, dist_size, INFINITY);

    EXPECT_EQ(*nearest_size, k);

    std::cout << "Points before sorting:" << std::endl;
    for (int i = 0; i < *nearest_size; i++) {
        std::cout << i << ". Point: " << Nearest_Points[i].x << ", " << Nearest_Points[i].y << ", "
                  << Nearest_Points[i].z << std::endl;
    }

    // Sort nearest points by distance to query (Bubblesort)
    for (int i = 0; i < *nearest_size; i++) {
        for (int j = 0; j < *nearest_size - i - 1; j++) {
            if (calc_dist(Nearest_Points[j], query) > calc_dist(Nearest_Points[j + 1], query)) {
                auto dummy = Nearest_Points[j];
                Nearest_Points[j] = Nearest_Points[j + 1];
                Nearest_Points[j + 1] = dummy;
            }
        }
    }

    std::cout << "Points after sorting:" << std::endl;

    for (int i = 0; i < *nearest_size; i++) {
        std::cout << i << ". Point: " << Nearest_Points[i].x << ", " << Nearest_Points[i].y << ", "
                  << Nearest_Points[i].z << std::endl;
    }

    // Check output
    EXPECT_NEAR(Nearest_Points[0].x, 1.1, 0.01);
    EXPECT_NEAR(Nearest_Points[1].x, -1.2, 0.01);
    EXPECT_NEAR(Nearest_Points[2].y, 1.3, 0.01);
    EXPECT_NEAR(Nearest_Points[3].y, -1.4, 0.01);
    EXPECT_NEAR(Nearest_Points[4].z, 1.5, 0.01);
    EXPECT_NEAR(Nearest_Points[5].z, -1.6, 0.01);

    cudaFree(managed_points);
    cudaFree(d_query);
    cudaFree(Nearest_Points);
    cudaFree(nearest_size);
    cudaFree(pointSearchSqDis);
    cudaFree(dist_size);
}
