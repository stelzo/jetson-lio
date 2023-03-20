#pragma once
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <vector>

#undef __CUDACC_VER__
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 0
#include <pcl/point_types.h>

#include <cuda_runtime.h>
#include <math/common.h>
#include <math/math.h>
#include <math/matrixXxX.h>

#include <use-ikfom.hpp>

struct EIGEN_ALIGN16 PointXYZINormal_CUDA
{
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
    union
    {
      struct
      {
        float intensity;
        float curvature;
      };
      float data_c[4];
    };
    PCL_MAKE_ALIGNED_OPERATOR_NEW

    EIGEN_DEVICE_FUNC  inline PointXYZINormal_CUDA (const PointXYZINormal_CUDA &p)
    {
      x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
      normal_x = p.normal_x; normal_y = p.normal_y; normal_z = p.normal_z; data_n[3] = 0.0f;
      curvature = p.curvature;
      intensity = p.intensity;
    }

   EIGEN_DEVICE_FUNC inline PointXYZINormal_CUDA ()
    {
      x = y = z = 0.0f;
      data[3] = 1.0f;
      normal_x = normal_y = normal_z = data_n[3] = 0.0f;
      intensity = 0.0f;
      curvature = 0;
    }
  
    EIGEN_DEVICE_FUNC friend std::ostream& operator << (std::ostream& os, const PointXYZINormal_CUDA& p);
};

int kf_point_state_step(
    void* _body_cloud, size_t body_cloud_size,
    void* _world_cloud, size_t world_cloud_size,
    void* _nearest_points, size_t nearest_points_size, size_t *nearest_points_sizes,
    void* _normvec, size_t normvec_size,
    state_ikfom *s,
    void *kd_tree,
    bool* point_selected_surf,
    void* _laser_cloud_ori, size_t laser_cloud_ori_size,
    void* _corr_normvect, size_t corr_normvect_size,
    bool converge);

void kf_jacobian(
    state_ikfom *s,
    bool* point_selected_surf,
    void* _laser_cloud_ori, size_t laser_cloud_ori_size,
    void* _corr_normvect, size_t corr_normvect_size,
    esekfom::dyn_share_datastruct<double> *ekfom_data, size_t effct_feat_num);

/** Only for testing */
void Raw_Nearest_Search(void* root, PointXYZINormal_CUDA* point, size_t k_nearest,
                    PointXYZINormal_CUDA* Nearest_Points, int *Nearest_Points_Size,
                    float* Point_Distance, size_t* Point_Distance_Size,
                    float max_dist);
void jacobian_test_cpu(double* data, Eigen::MatrixXd* target, int i);
void jacobian_test_gpu(double* data, int rows, int cols, Eigen::MatrixXd& target, int i);

bool pca_custom_iterative(const rmagine::Vector3d* cluster, size_t cluster_size, float normal_threshold, rmagine::Vector3d* normal, rmagine::VectorN<4, double>* plane_equation);
bool pca_constant(const rmagine::Vector3d* cluster, size_t cluster_size, float normal_threshold, rmagine::Vector3d* normal, rmagine::VectorN<4, double>* plane_equation);

