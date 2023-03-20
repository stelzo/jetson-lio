#include "ikd_Tree_gpu_search.h"
#include <cuda_runtime_api.h>

#undef __CUDACC_VER__
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 0

#include <Eigen/Eigen>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <cuda-utils.h>
#include <pcl/point_types.h>

#include <cusolverDn.h>

#include <use-ikfom.hpp>
#include <disable_stupid_warnings.h>

#include <math/common.h>
#include <math/math.h>
#include <math/matrixXxX.h>


#define K_NEAREST = 5; // TODO still hardcoded in kernels

template <typename PointType>
struct KD_TREE_NODE {
    PointType point;
    int division_axis;
    int TreeSize = 1;
    int invalid_point_num = 0;
    int down_del_num = 0;
    bool point_deleted = false;
    bool tree_deleted = false;
    bool point_downsample_deleted = false;
    bool tree_downsample_deleted = false;
    bool need_push_down_to_left = false;
    bool need_push_down_to_right = false;
    bool working_flag = false;
    pthread_mutex_t push_down_mutex_lock;
    float node_range_x[2], node_range_y[2], node_range_z[2];
    float radius_sq;
    KD_TREE_NODE *left_son_ptr = nullptr;
    KD_TREE_NODE *right_son_ptr = nullptr;
    KD_TREE_NODE *father_ptr = nullptr;
    // For paper data record
    float alpha_del;
    float alpha_bal;
};

struct PointType_CMP_GPU
{
    PointXYZINormal_CUDA point;
    float dist = 0.0;
 
    __device__ PointType_CMP_GPU()
    {
        point.x = 0;
        point.y = 0;
        point.z = 0;
        dist = 0.0;
    }

    __device__ PointType_CMP_GPU(PointXYZINormal_CUDA p, float d = INFINITY)
    {
        this->point = p;
        this->dist = d;
    };

    __device__ bool operator<(const PointType_CMP_GPU &a) const
    {
        if (fabs(dist - a.dist) < 1e-10)
            return point.x < a.point.x;
        else
            return dist < a.dist;
    }
};

struct MANUAL_HEAP_GPU
{
    PointType_CMP_GPU heap[10]; // knearest * 2
    int heap_size;
    int cap;

    __device__ MANUAL_HEAP_GPU()
    {
        init();
    }

    __device__ void init()
    {
        heap_size = 0;
        cap = 10;
    }

    __device__ void pop()
    {
        if (heap_size == 0)
            return;
        heap[0] = heap[heap_size - 1];
        heap_size--;
        MoveDown(0);
        return;
    }
    
    __device__ PointType_CMP_GPU top()
    {
        return heap[0];
    }
    
    __device__ void push(PointType_CMP_GPU point)
    {
        if (heap_size >= cap)
            return;        
        heap[heap_size] = point;
        FloatUp(heap_size);
        heap_size++;
        return;
    }
    
    __device__ int size()
    {
        return heap_size;
    }

    __device__ void clear()
    {
        heap_size = 0;
        return;
    }
    
    __device__ void MoveDown(int heap_index)
    {
        int l = heap_index * 2 + 1;
        PointType_CMP_GPU tmp = heap[heap_index];
        while (l < heap_size)
        {
            if (l + 1 < heap_size && heap[l] < heap[l + 1])
                l++;
            if (tmp < heap[l])
            {
                heap[heap_index] = heap[l];
                heap_index = l;
                l = heap_index * 2 + 1;
            }
            else
                break;
        }
        heap[heap_index] = tmp;
        return;
    }

    __device__ void FloatUp(int heap_index)
    {
        int ancestor = (heap_index - 1) / 2;
        PointType_CMP_GPU tmp = heap[heap_index];
        while (heap_index > 0)
        {
            if (heap[ancestor] < tmp)
            {
                heap[heap_index] = heap[ancestor];
                heap_index = ancestor;
                ancestor = (heap_index - 1) / 2;
            }
            else
                break;
        }
        heap[heap_index] = tmp;
        return;
    }
};

JLIO_INLINE_FUNCTION float
calc_dist_gpu(PointXYZINormal_CUDA a, PointXYZINormal_CUDA b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}


__device__
float calc_box_dist_gpu(KD_TREE_NODE<PointXYZINormal_CUDA> *node, PointXYZINormal_CUDA point)
{
    if (node == nullptr)
        return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0])
    {
        min_dist += (point.x - node->node_range_x[0]) * (point.x - node->node_range_x[0]);
    }
    if (point.x > node->node_range_x[1])
    {
        min_dist += (point.x - node->node_range_x[1]) * (point.x - node->node_range_x[1]);
    }
    if (point.y < node->node_range_y[0])
    {
        min_dist += (point.y - node->node_range_y[0]) * (point.y - node->node_range_y[0]);
    }
    if (point.y > node->node_range_y[1])
    {
        min_dist += (point.y - node->node_range_y[1]) * (point.y - node->node_range_y[1]);
    }
    if (point.z < node->node_range_z[0])
    {
        min_dist += (point.z - node->node_range_z[0]) * (point.z - node->node_range_z[0]);
    }
    if (point.z > node->node_range_z[1])
    {
        min_dist += (point.z - node->node_range_z[1]) * (point.z - node->node_range_z[1]);
    }

    /*if (min_dist < 0.001)
    {
        printf("min_dist < %f, node_range_x %f. %f. %f, node_range_y %f. %f. %f, node_range_z %f. %f. %f actual p: %f, %f, %f, PPDIST=%f\n",
        min_dist, node->node_range_x[0], node->node_range_x[1], point.x,
        node->node_range_y[0], node->node_range_y[1], point.y,
         node->node_range_z[0], node->node_range_z[1], point.z,
         node->point.x, node->point.y, node->point.z,
         calc_dist_gpu(node->point, point));
    }*/
    return min_dist;
}


__device__ void Push_Down(KD_TREE_NODE<PointXYZINormal_CUDA> *root) {
    if (root == nullptr) return;

    if (root->need_push_down_to_left && root->left_son_ptr != nullptr) {
        root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
        root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
        root->left_son_ptr->tree_deleted =
            root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
        root->left_son_ptr->point_deleted =
            root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
        if (root->tree_downsample_deleted)
            root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
        if (root->tree_deleted)
            root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
        else
            root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;
        root->left_son_ptr->need_push_down_to_left = true;
        root->left_son_ptr->need_push_down_to_right = true;
        root->need_push_down_to_left = false;
        
    }
    if (root->need_push_down_to_right && root->right_son_ptr != nullptr) {
        root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
        root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
        root->right_son_ptr->tree_deleted =
            root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
        root->right_son_ptr->point_deleted =
            root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
        if (root->tree_downsample_deleted)
            root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
        if (root->tree_deleted)
            root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
        else
            root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;
        root->right_son_ptr->need_push_down_to_left = true;
        root->right_son_ptr->need_push_down_to_right = true;
        root->need_push_down_to_right = false;
    }
    return;
}
__device__
void Search(KD_TREE_NODE<PointXYZINormal_CUDA> *root, int k_nearest, PointXYZINormal_CUDA point, MANUAL_HEAP_GPU* q, float max_dist)
{
    if (root == nullptr || root->tree_deleted)
    {
        return;
    }

    float max_dist_sqr = max_dist * max_dist;
    float dist_to_kth_neighbor = FLT_MAX;

    bool bt = false;
    KD_TREE_NODE<PointXYZINormal_CUDA>* prev_node = nullptr;
    KD_TREE_NODE<PointXYZINormal_CUDA>* current_node = root;

    do
    {
        if (root->tree_deleted)
        {
            break;
        }

        float cur_dist = calc_box_dist_gpu(current_node, point);
    
        if (current_node->need_push_down_to_left || current_node->need_push_down_to_right)
        {
            Push_Down(current_node);
        }

        if (!bt && !current_node->point_deleted)
        {
            float dist = calc_dist_gpu(point, current_node->point);
            //printf("dist: %f %f %f %f\n", current_node->point.x, current_node->point.y, current_node->point.z, dist);
            if (dist <= max_dist_sqr && (q->size() < k_nearest || dist < q->top().dist))
            {
                if (q->size() >= k_nearest)
                    q->pop();
                PointType_CMP_GPU current_point(current_node->point, dist);
                q->push(current_point);
                //printf("push point: %f %f %f %f\n", current_node->point.x, current_node->point.y, current_node->point.z, dist);
                if (q->size() == k_nearest) {
                    dist_to_kth_neighbor = q->top().dist;
                }
            }
        }

        auto* child_left = current_node->left_son_ptr;
        auto* child_right = current_node->right_son_ptr;

        float dist_left_node = calc_box_dist_gpu(child_left, point);
        float dist_right_node = calc_box_dist_gpu(child_right, point);

        bool traverse_left = child_left != nullptr && dist_left_node <= min(dist_to_kth_neighbor, max_dist_sqr);
        bool traverse_right = child_right != nullptr && dist_right_node <= min(dist_to_kth_neighbor, max_dist_sqr);

        auto* best_child = (dist_left_node <= dist_right_node) ? child_left : child_right;
        auto* other_child = (dist_left_node <= dist_right_node) ? child_right : child_left;

        if (!bt) {
            if (!traverse_left && !traverse_right) {
                bt = true;
                auto parent = current_node->father_ptr;
                prev_node = current_node;
                current_node = parent;
            } else {
                prev_node = current_node;
                current_node = (traverse_left) ? child_left : child_right;
                if (traverse_left && traverse_right) {
                    current_node = best_child;
                }
            }
        } else {
            float mind(INFINITY);

            if (other_child != nullptr) {
                mind = max(dist_left_node, dist_right_node);
            }

            if (other_child != nullptr && prev_node == best_child && mind <= dist_to_kth_neighbor) {
                prev_node = current_node;
                current_node = other_child;
                bt = false;
            } else {
                auto parent = current_node->father_ptr;
                prev_node = current_node;
                current_node = parent;
            }
        }
    } while (current_node != nullptr);
}


__device__
void Search_recursive(KD_TREE_NODE<PointXYZINormal_CUDA> *root, int k_nearest, PointXYZINormal_CUDA point, MANUAL_HEAP_GPU* q, float max_dist)
{
    if (root == nullptr || root->tree_deleted)
    {
        return;
    }

    float cur_dist = calc_box_dist_gpu(root, point);
    float max_dist_sqr = max_dist * max_dist;
    if (cur_dist > max_dist_sqr)
    {
        return;        
    }
    
    if (root->need_push_down_to_left || root->need_push_down_to_right)
    {
        printf("ERR: PUSH DOWN ON GPU SHOULD NOT BE TRIGGERED\n");
    }

    if (!root->point_deleted)
    {
        float dist = calc_dist_gpu(point, root->point);
        if (dist <= max_dist_sqr && (q->size() < k_nearest || dist < q->top().dist))
        {
            //printf("found a point dist %f, maxdist %f, topdist %f\n", dist, max_dist_sqr, q->top().dist);
            if (q->size() >= k_nearest)
                q->pop();
            PointType_CMP_GPU current_point(root->point, dist);
            q->push(current_point);
        }
    }

    float dist_right_node = calc_box_dist_gpu(root->right_son_ptr, point);
    float dist_left_node = calc_box_dist_gpu(root->left_son_ptr, point);
    if (q->size() < k_nearest || dist_left_node < q->top().dist && dist_right_node < q->top().dist)
    {
        if (dist_left_node <= dist_right_node)
        {
            Search(root->left_son_ptr, k_nearest, point, q, max_dist);

            if (q->size() < k_nearest || dist_right_node < q->top().dist)
            {
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);
            }
        }
        else
        {
            Search(root->right_son_ptr, k_nearest, point, q, max_dist);

            if (q->size() < k_nearest || dist_left_node < q->top().dist)
            {
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);
            }
        }
    }
    else
    {
        if (dist_left_node < q->top().dist)
        {
            Search(root->left_son_ptr, k_nearest, point, q, max_dist);
        }
        if (dist_right_node < q->top().dist)
        {
            Search(root->right_son_ptr, k_nearest, point, q, max_dist);
        }
    }
    return;
}

__device__
void Nearest_Search(void* root, PointXYZINormal_CUDA point, size_t k_nearest,
                    PointXYZINormal_CUDA* Nearest_Points, int *Nearest_Points_Size,
                    float* Point_Distance, size_t* Point_Distance_Size,
                    float max_dist
                    )
{
    MANUAL_HEAP_GPU q;
    q.init();

    KD_TREE_NODE<PointXYZINormal_CUDA>* tree = (KD_TREE_NODE<PointXYZINormal_CUDA>*) root;
    Search(tree, k_nearest, point, &q, max_dist);

    int k_found = min((int)k_nearest, int(q.size()));
    *Point_Distance_Size = 0;
    *Nearest_Points_Size = 0;
    for (int i = 0; i < k_found; i++)
    {
        Nearest_Points[*Nearest_Points_Size] = q.top().point;
        Point_Distance[*Point_Distance_Size] = q.top().dist;
        q.pop();

        (*Nearest_Points_Size)++;
        (*Point_Distance_Size)++;
    }

    //(*Nearest_Points_Size)--;
    //(*Point_Distance_Size)--;

    /*printf("search done. k-found: %d, 1st nearest p %f, %f, %f; dist %f\n", 
    k_found, Nearest_Points[*Nearest_Points_Size-1].x, Nearest_Points[*Nearest_Points_Size-1].y, Nearest_Points[*Nearest_Points_Size-1].z,
    Point_Distance[*Point_Distance_Size-1]);*/

    //printf("search done. k-found: %d\n", k_found);
    return;
}

__global__ void krnl_raw_nearest_search(void* root, PointXYZINormal_CUDA* point, size_t k_nearest,
                    PointXYZINormal_CUDA* Nearest_Points, int *Nearest_Points_Size,
                    float* Point_Distance, size_t* Point_Distance_Size,
                    float max_dist)
{
    Nearest_Search(root, *point, k_nearest, Nearest_Points, Nearest_Points_Size, Point_Distance, Point_Distance_Size, max_dist);
}

void Raw_Nearest_Search(void* root, PointXYZINormal_CUDA* point, size_t k_nearest,
                    PointXYZINormal_CUDA* Nearest_Points, int *Nearest_Points_Size,
                    float* Point_Distance, size_t* Point_Distance_Size,
                    float max_dist) 
{
    krnl_raw_nearest_search<<<1, 1>>>(root, point, k_nearest, Nearest_Points, Nearest_Points_Size, Point_Distance, Point_Distance_Size, max_dist);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
}

template<typename T, int Rows>
__device__ Eigen::Matrix<T, Rows, 1> mean_matrix(const T *data, const size_t size)
{
    Eigen::Matrix<T, Rows, 1> mean = Eigen::Matrix<T, Rows, 1>::Zero();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < Rows; j++) {
            mean(j) += data[i * Rows + j];
        }
    }
    mean /= size;
    return mean;
}

template<typename T, int Rows>
__device__ Eigen::Matrix<T, Rows, Rows> covariance_matrix(const T *data, const size_t size, const Eigen::Matrix<T, Rows, 1> &mean)
{
    Eigen::Matrix<T, Rows, Rows> covariance = Eigen::Matrix<T, Rows, Rows>::Zero();
    for (int i = 0; i < size; i++) {
        Eigen::Matrix<T, Rows, 1> diff = Eigen::Map<const Eigen::Matrix<T, Rows, 1>>(data + i * Rows) - mean;
        covariance += diff * diff.transpose();
    }
    covariance /= size;
    return covariance;
}

template<typename T, int Rows>
__device__ void eigen_decomposition(const Eigen::Matrix<T, Rows, Rows> &matrix, Eigen::Matrix<T, Rows, Rows> &eigenvectors, Eigen::Matrix<T, Rows, 1> &eigenvalues)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Rows, Rows>> solver(matrix);
    eigenvectors = solver.eigenvectors();
    eigenvalues = solver.eigenvalues();
}



template<typename T>
__device__ bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, Eigen::Matrix<T, 3, 1>* points, size_t points_size, const T threshold)
{
    constexpr size_t NUM_MATCH_POINTS = 5;
    constexpr size_t MIN_NUM_MATCH_POINTS = 3;
    
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    A.setZero();

    if (points_size < MIN_NUM_MATCH_POINTS)
    {
        return false;
    }

    for (int j = 0; j < min((int)NUM_MATCH_POINTS, (int)points_size); j++)
    {
        A(j,0) = points[j].x();
        A(j,1) = points[j].y();
        A(j,2) = points[j].z();
    }

    Eigen::Matrix<T, 3, NUM_MATCH_POINTS> A_transpose = A.transpose();
    Eigen::Matrix<T, 3, 3> covariance = A_transpose * A;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> eig(covariance);
    const auto& eigenvector = eig.eigenvectors().col(0);

    T n = eigenvector.norm();
    pca_result(0) = eigenvector(0) / n;
    pca_result(1) = eigenvector(1) / n;
    pca_result(2) = eigenvector(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < min((int)NUM_MATCH_POINTS, (int)points_size); j++)
    {
        if (fabs(pca_result(0) * points[j].x() + pca_result(1) * points[j].y() + pca_result(2) * points[j].z() + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

__global__ void krnl_jacobian(  PointXYZINormal_CUDA* laser_cloud_ori, size_t laser_cloud_ori_size,
                                PointXYZINormal_CUDA* corr_normvect, size_t corr_normvect_size,
                                rmagine::Quaterniond rot,
                                rmagine::Vector3d offset_T_L_I,
                                rmagine::Quaterniond offset_R_L_I,
                                double *h_x_raw, int h_x_rows, int h_x_cols,
                                double *h_raw, int h_rows, int h_cols,
                                bool extrinsic_est_en)
{
    // ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    //ekfom_data.h.resize(effct_feat_num);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= laser_cloud_ori_size)
        return;

    rmagine::Vector3d v(laser_cloud_ori[i].x, laser_cloud_ori[i].y, laser_cloud_ori[i].z);
    rmagine::Matrix3x3d S;
    S(0, 0) = 0.0; S(0, 1) = -v.z; S(0, 2) = v.y;
    S(1, 0) = v.z; S(1, 1) = 0.0; S(1, 2) = -v.x;
    S(2, 0) = -v.y; S(2, 1) = v.x; S(2, 2) = 0.0;
    rmagine::Vector3d point_this = offset_R_L_I * v + offset_T_L_I;

    /*** get the normal vector of closest surface/corner ***/
    const PointXYZINormal_CUDA &norm_p = corr_normvect[i];
    rmagine::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    rmagine::Vector3d C = rot.conjugate() * norm_vec;
    rmagine::Vector3d A = S * C;
    if (extrinsic_est_en) // false with given extrinsic
    {
        rmagine::Vector3d B = S * (offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
        h_x_raw[i + 0 * h_x_cols] = norm_p.x;
        h_x_raw[i + 1 * h_x_cols] = norm_p.y;
        h_x_raw[i + 2 * h_x_cols] = norm_p.z;

        h_x_raw[i + 3 * h_x_cols] = A.x;
        h_x_raw[i + 4 * h_x_cols] = A.y;
        h_x_raw[i + 5 * h_x_cols] = A.z;

        h_x_raw[i + 6 * h_x_cols] = B.x;
        h_x_raw[i + 7 * h_x_cols] = B.y;
        h_x_raw[i + 8 * h_x_cols] = B.z;

        h_x_raw[i + 9 * h_x_cols] = C.x;
        h_x_raw[i + 10 * h_x_cols] = C.y;
        h_x_raw[i + 11 * h_x_cols] = C.z;
    }
    else
    {
        h_x_raw[i + 0 * h_x_cols] = norm_p.x;
        h_x_raw[i + 1 * h_x_cols] = norm_p.y;
        h_x_raw[i + 2 * h_x_cols] = norm_p.z;

        h_x_raw[i + 3 * h_x_cols] = A.x;
        h_x_raw[i + 4 * h_x_cols] = A.y;
        h_x_raw[i + 5 * h_x_cols] = A.z;

        h_x_raw[i + 6 * h_x_cols] = 0.0;
        h_x_raw[i + 7 * h_x_cols] = 0.0;
        h_x_raw[i + 8 * h_x_cols] = 0.0;

        h_x_raw[i + 9 * h_x_cols] = 0.0;
        h_x_raw[i + 10 * h_x_cols] = 0.0;
        h_x_raw[i + 11 * h_x_cols] = 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    h_raw[i] = -norm_p.intensity;
}

RMAGINE_INLINE_FUNCTION
rmagine::Vector3d calc_centroid(const rmagine::Vector3d* cluster, size_t cluster_size)
{
    rmagine::Vector3d sum(0, 0, 0);
    for (size_t i = 0; i < cluster_size; i++)
    {
        sum.addInplace(cluster[i]);
    }

    return sum * (1.0 / static_cast<double>(cluster_size));
}

RMAGINE_INLINE_FUNCTION
rmagine::Matrix3x3d calc_cov(const rmagine::Vector3d* cluster, size_t cluster_size, const rmagine::Vector3d& centroid)
{
    double xx = 0.0; double xy = 0.0; double xz = 0.0;
    double yy = 0.0; double yz = 0.0; double zz = 0.0;

    for (size_t i = 0; i < cluster_size; i++)
    {
        rmagine::Vector3d r = cluster[i] - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= static_cast<double>(cluster_size);
    xy /= static_cast<double>(cluster_size);
    xz /= static_cast<double>(cluster_size);
    yy /= static_cast<double>(cluster_size);
    yz /= static_cast<double>(cluster_size);
    zz /= static_cast<double>(cluster_size);

    rmagine::Matrix3x3d cov;
    cov(0, 0) = xx;
    cov(0, 1) = xy;
    cov(0, 2) = xz;
    cov(1, 0) = xy;
    cov(1, 1) = yy;
    cov(1, 2) = yz;
    cov(2, 0) = xz;
    cov(2, 1) = yz;
    cov(2, 2) = zz;

    return cov;
}

// https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
__host__ __device__ bool pca_constant(const rmagine::Vector3d* cluster, size_t n, float normal_threshold, rmagine::Vector3d* normal, rmagine::VectorN<4, double>* plane_equation) {
    if (n < 3)
    {
        return false;
    }

    rmagine::Vector3d centroid = calc_centroid(cluster, n);
    rmagine::Matrix3x3d cov = calc_cov(cluster, n, centroid);

    double xx = cov(0, 0); double xy = cov(0, 1); double xz = cov(0, 2);
    double yy = cov(1, 1); double yz = cov(1, 2); double zz = cov(2, 2);

    rmagine::Vector3d weighted_dir(0, 0, 0);

    {
        double det_x = yy*zz - yz*yz;
        rmagine::Vector3d axis_dir(det_x, xz*yz - xy*zz, xy*yz - xz*yy);
        double weight = det_x * det_x;
        if (weighted_dir.dot(axis_dir) < 0.0)
        {
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    {
        double det_y = xx*zz - xz*xz;
        rmagine::Vector3d axis_dir(xz*yz - xy*zz, det_y, xy*xz - yz*xx);
        double weight = det_y * det_y;
        if (weighted_dir.dot(axis_dir) < 0.0)
        { 
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    {
        double det_z = xx*yy - xy*xy;
        rmagine::Vector3d axis_dir(xy*yz - xz*yy, xy*xz - yz*xx, det_z);
        double weight = det_z * det_z;
        if (weighted_dir.dot(axis_dir) < 0.0)
        { 
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    *normal = weighted_dir.normalized();

    double d = -(normal->x * centroid.x + normal->y * centroid.y + normal->z * centroid.z);

    (*plane_equation)(0) = (*normal).x;
    (*plane_equation)(1) = (*normal).y;
    (*plane_equation)(2) = (*normal).z;
    (*plane_equation)(3) = d;

    // if any of the points is too far from the plane, return false
    for (size_t i = 0; i < n; i++)
    {
        if (fabs((*plane_equation)(0) * cluster[i].x + (*plane_equation)(1) * cluster[i].y + (*plane_equation)(2) * cluster[i].z + (*plane_equation)(3)) > normal_threshold)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief estimate the plane equation
 * @param pca_result: the plane equation
 * @param point: the point cloud
 * @param threshold: the threshold of the distance between the point and the plane
 * @return true if the plane equation is estimated successfully
 * @note the plane equation: Ax + By + Cz + D = 0
 *      convert to: A/D*x + B/D*y + C/D*z = -1
 *     solve: A0*x0 = b0
 *    where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
 *   normvec:  normalized x0
*/
__host__ __device__ bool estimate_plane(rmagine::VectorN<4, double> &pca_result, const rmagine::Vector3d *point, const size_t n, const float threshold)
{
    return true;
}

/**
 * Plane Normal vector estimation using PCA
 * 
*/
__host__ __device__ bool pca_custom_iterative(const rmagine::Vector3d* cluster, size_t cluster_size, float normal_threshold, rmagine::Vector3d* normal, rmagine::VectorN<4, double>* plane_equation)
{
    if (cluster_size < 3)
    {
        return false;
    }

    rmagine::Vector3d cog = calc_centroid(cluster, cluster_size);
    rmagine::Matrix3x3d K = calc_cov(cluster, cluster_size, cog);

    double A[3][3] = {{K(0, 0), K(0, 1), K(0, 2)}, {K(1, 0), K(1, 1), K(1, 2)}, {K(2, 0), K(2, 1), K(2, 2)}};

    // jacobi eigen solver
    // init eigen vector matrix as identity
    double eigenvectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double eigenvalues[3] = {0, 0, 0};

    int n = 3;

    // Compute the off-diagonal norm of A
    double offdiag_norm = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            offdiag_norm += A[i][j] * A[i][j];
        }
    }

    // Compute the eigenvalues and eigenvectors
    int max_iter = 1000;
    for (int iter = 0; iter < max_iter; iter++) {
        // Find the largest off-diagonal element
        int p = 0;
        int q = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i+1; j < n; j++) {
                if (fabs(A[i][j]) > fabs(A[p][q])) {
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        double threshold = 1e-8;
        if (offdiag_norm < threshold) {
            break;
        }

        // Compute the Jacobi rotation angle
        double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        double t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0) {
            t = -t;
        }

        // Compute the Jacobi rotation matrix
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;

        // Apply the Jacobi rotation
        double A_pq = A[p][q];
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        A[p][p] -= t * A_pq;
        A[q][q] += t * A_pq;
        for (int r = 0; r < n; r++) {
            if (r != p && r != q) {
                double A_pr = A[p][r];
                double A_qr = A[q][r];
                A[p][r] = c * A_pr - s * A_qr;
                A[r][p] = A[p][r];
                A[q][r] = c * A_qr + s * A_pr;
                A[r][q] = A[q][r];
            }

            // Update the eigenvectors
            double eigenvector_pr = eigenvectors[p][r];
            double eigenvector_qr = eigenvectors[q][r];
            eigenvectors[p][r] = c * eigenvector_pr - s * eigenvector_qr;
            eigenvectors[q][r] = c * eigenvector_qr + s * eigenvector_pr;
        }

        // Update the off-diagonal norm
        offdiag_norm -= A_pq * A_pq;
    }

    // Sort the eigenvalues and eigenvectors
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (eigenvalues[i] > eigenvalues[j]) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
                for (int k = 0; k < n; k++) {
                    double tmp = eigenvectors[k][i];
                    eigenvectors[k][i] = eigenvectors[k][j];
                    eigenvectors[k][j] = tmp;
                }
            }
        }
    }


    //printf("eigenvalues: %lf %lf %lf \n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);

    // eigen vector with smallest eigen value is the normal
    double minEigenV[3] = {eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]};

    // normalize
    double norm = sqrt(minEigenV[0] * minEigenV[0] + minEigenV[1] * minEigenV[1] + minEigenV[2] * minEigenV[2]);
    minEigenV[0] /= norm;
    minEigenV[1] /= norm;
    minEigenV[2] /= norm;
    double d = -(minEigenV[0] * cog.x + minEigenV[1] * cog.y + minEigenV[2] * cog.z);

    (*plane_equation)(0) = minEigenV[0];
    (*plane_equation)(1) = minEigenV[1];
    (*plane_equation)(2) = minEigenV[2];
    (*plane_equation)(3) = d;

    normal->x = (float)minEigenV[0];
    normal->y = (float)minEigenV[1];
    normal->z = (float)minEigenV[2];

    for (size_t i = 0; i < cluster_size; i++)
    {
        double x = minEigenV[0] * cluster[i].x;
        double y = minEigenV[1] * cluster[i].y;
        double z = minEigenV[2] * cluster[i].z;

        double absolute = fabs(x+y+z);
        if (absolute > normal_threshold)
        {
            return false;
        }
    }

    return true;
}

__global__ void krnl_point_kf_state(void* _body_cloud, size_t body_cloud_size,
                               void* _world_cloud, size_t world_cloud_size,
                               void* _nearest_points, size_t nearest_points_size, size_t *nearest_points_sizes,
                               void* _normvec, size_t normvec_size,
                               void *kd_tree,
                               bool* point_selected_surf,
                               bool ekfom_data_converged,
                               SO3 rot,
                            vect3 offset_T_L_I,
                            SO3 offset_R_L_I,
                            vect3 pos
                               )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_cloud_size)
        return;

    // conversion to cuda types with same memory layout
    PointXYZINormal_CUDA* body_cloud = (PointXYZINormal_CUDA*)_body_cloud;
    PointXYZINormal_CUDA* world_cloud = (PointXYZINormal_CUDA*)_world_cloud;
    PointXYZINormal_CUDA* nearest_points = (PointXYZINormal_CUDA*)_nearest_points;
    PointXYZINormal_CUDA* normvec = (PointXYZINormal_CUDA*)_normvec;

    constexpr size_t NUM_MATCH_POINTS = 5; // hardcoded everywhere but defines are not allowed in device code

    point_selected_surf[i] = false; // initialize if point is relevant for surface feature

    // map frame conversion with last pose
    Eigen::Vector3d p_body(body_cloud[i].x, body_cloud[i].y, body_cloud[i].z);
    Eigen::Vector3d p_global(rot * (offset_R_L_I * p_body + offset_T_L_I) + pos);
    world_cloud[i].x = p_global(0);
    world_cloud[i].y = p_global(1);
    world_cloud[i].z = p_global(2);
    world_cloud[i].intensity = body_cloud[i].intensity;

    // closest distances to map points
    float pointSearchSqDis[NUM_MATCH_POINTS];
    size_t dist_size = NUM_MATCH_POINTS;

    if (ekfom_data_converged)
    {
        // find closest points in map
        int nearest_size = 0;
        Nearest_Search(kd_tree, world_cloud[i], NUM_MATCH_POINTS, &nearest_points[i*NUM_MATCH_POINTS], &nearest_size, pointSearchSqDis, &dist_size, INFINITY);
        nearest_points_sizes[i] = (size_t)nearest_size;
        
        point_selected_surf[i] = nearest_points_sizes[i] < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
    }
    
    // disregard points that are not relevant for surface feature because they are too far away from every map point
    if (!point_selected_surf[i])
    {
        return;
    }

    point_selected_surf[i] = false;

    // build array of Eigen vectors for plane estimation
    rmagine::Vector3d nearest_points_custom[NUM_MATCH_POINTS];
    for (int j = 0; j < nearest_points_sizes[i]; j++)
    {
        nearest_points_custom[j] =  rmagine::Vector3d(nearest_points[i*NUM_MATCH_POINTS + j].x,
                                    nearest_points[i*NUM_MATCH_POINTS + j].y, 
                                    nearest_points[i*NUM_MATCH_POINTS + j].z);
    }

    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> plane_coeffs;

    bool found_normal = pca_constant(nearest_points_custom, nearest_points_sizes[i], (double)0.1f, &normal, &plane_coeffs);
    if (!found_normal)
    {
        return;
    }

    normvec[i].x = (float) plane_coeffs(0);
    normvec[i].y = (float) plane_coeffs(1);
    normvec[i].z = (float) plane_coeffs(2);
    normvec[i].curvature = (float) plane_coeffs(3);

    Eigen::Vector3d delta = p_global - pos;
    if (normvec[i].x * delta(0) + normvec[i].y * delta(1) + normvec[i].z * delta(2) > 0.0f)
    {
        normvec[i].x = -normvec[i].x;
        normvec[i].y = -normvec[i].y;
        normvec[i].z = -normvec[i].z;
    }

    float pd2 = normvec[i].x * world_cloud[i].x + normvec[i].y * world_cloud[i].y + normvec[i].z * world_cloud[i].z + normvec[i].curvature;
    double p_body_norm = sqrtf(p_body(0) * p_body(0) + p_body(1) * p_body(1) + p_body(2) * p_body(2));
    double s = 1.0f - 0.9f * fabs(pd2) / p_body_norm;

    point_selected_surf[i] = fabs(s) > 0.9f;
}

__global__ void krnl_filter_selected_surf(   PointXYZINormal_CUDA* body_cloud, size_t body_cloud_size,
                                        PointXYZINormal_CUDA* normvec, size_t normvec_size,
                                        bool* point_selected_surf,
                                        PointXYZINormal_CUDA* laser_cloud_ori, size_t laser_cloud_ori_size,
                                        PointXYZINormal_CUDA* corr_normvect, size_t corr_normvect_size,
                                        int* effct_feat_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_cloud_size || i >= normvec_size)
    {
        return;
    }

    if (!point_selected_surf[i])
    {
        //printf("point selected surf false\n");
        return;
   }

    int old_effct_feat_num = atomicAdd(effct_feat_num, 1);
    //printf("effct num %d\n", old_effct_feat_num);
    laser_cloud_ori[old_effct_feat_num] = body_cloud[i];
    corr_normvect[old_effct_feat_num] = normvec[i];
}


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
    bool converge)
{
    PointXYZINormal_CUDA* body_cloud = (PointXYZINormal_CUDA*)_body_cloud;
    PointXYZINormal_CUDA* world_cloud = (PointXYZINormal_CUDA*)_world_cloud;
    PointXYZINormal_CUDA* nearest_points = (PointXYZINormal_CUDA*)_nearest_points;
    PointXYZINormal_CUDA* normvec = (PointXYZINormal_CUDA*)_normvec;
    PointXYZINormal_CUDA* laser_cloud_ori = (PointXYZINormal_CUDA*)_laser_cloud_ori;
    PointXYZINormal_CUDA* corr_normvect = (PointXYZINormal_CUDA*)_corr_normvect;

    constexpr size_t THREADS_PER_BLOCK = 1024;

    //std::cout << "body_cloud_size " << body_cloud_size << ", normvec_size " << normvec_size << std::endl;
    size_t kf_state_grid_dim = static_cast<size_t>(std::ceil((float)body_cloud_size / THREADS_PER_BLOCK));

    krnl_point_kf_state<<<kf_state_grid_dim, THREADS_PER_BLOCK>>>(
        _body_cloud, body_cloud_size,
        _world_cloud, world_cloud_size,
        _nearest_points, nearest_points_size, nearest_points_sizes,
        _normvec, normvec_size,
        kd_tree,
        point_selected_surf,
        converge,
        s->rot,
        s->offset_T_L_I,
        s->offset_R_L_I,
        s->pos        
        );
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    //std::cout << "point_kf_state run successfull" << std::endl;

    size_t selected_surf_grid_dim = static_cast<size_t>(std::ceil((float)body_cloud_size / THREADS_PER_BLOCK));
    //std::cout << "Dim " << selected_surf_grid_dim << ", " << THREADS_PER_BLOCK << std::endl;

    CHECK_LAST_CUDA_ERROR();
    int *effct_feat_num = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&effct_feat_num, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(effct_feat_num, 0, sizeof(int)));

    int *effct_feat_num_host = nullptr;
    effct_feat_num_host = new int;

    krnl_filter_selected_surf<<<selected_surf_grid_dim, THREADS_PER_BLOCK>>>(
        body_cloud, body_cloud_size,
        normvec, normvec_size,
        point_selected_surf,
        laser_cloud_ori, laser_cloud_ori_size,
        corr_normvect, corr_normvect_size,
        effct_feat_num
    );
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cudaMemcpy(effct_feat_num_host, effct_feat_num, sizeof(int), cudaMemcpyDeviceToHost);

    if (effct_feat_num)
    {
        CHECK_CUDA_ERROR(cudaFree(effct_feat_num));
        effct_feat_num = nullptr;
    }

    int res = *effct_feat_num_host;
    if (effct_feat_num_host)
    {
        delete effct_feat_num_host;
    }

    return res;
}

__global__
void krnl_jacobian_test(double* data, double* mat, int rows, int cols, int i)
{
    for(size_t j = 0; j < 12; j++)
    {
        mat[i + j * rows] = data[j];
    }
}

void jacobian_test_cpu(double* data, Eigen::MatrixXd* target, int i)
{
    assert(target != nullptr);
    assert(target->data() != nullptr);

    int h_x_rows = target->rows();
    int h_x_cols = target->cols();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> h_x(target->data(), h_x_rows, h_x_cols);
    h_x.block<1, 12>(i, 0) << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11];
}

void jacobian_test_gpu(double* data, int rows, int cols, Eigen::MatrixXd& target, int i)
{
    assert(data != nullptr);

    rmagine::MatrixXd mdata_host(rows, cols);

    double* data_gpu = nullptr;
    cudaMalloc(&data_gpu, sizeof(double) * 12);
    cudaMemcpy(data_gpu, data, sizeof(double) * 12, cudaMemcpyHostToDevice);

    krnl_jacobian_test<<<1, 1>>>(data_gpu, mdata_host.m_data, rows, cols, i);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    assert(mdata_host(i, 2) > 1.1);
    assert(mdata_host(i, 2) < 1.3);
    assert(mdata_host(i, 3) > 1.2);

    mdata_host.toEigenInpl(target);

    cudaFree(data_gpu);
}

void kf_jacobian(
    state_ikfom *s,
    bool* point_selected_surf,
    void* _laser_cloud_ori, size_t laser_cloud_ori_size,
    void* _corr_normvect, size_t corr_normvect_size,
    esekfom::dyn_share_datastruct<double> *ekfom_data, size_t effct_feat_num)
{
    PointXYZINormal_CUDA* laser_cloud_ori = (PointXYZINormal_CUDA*)_laser_cloud_ori;
    PointXYZINormal_CUDA* corr_normvect = (PointXYZINormal_CUDA*)_corr_normvect;
    constexpr size_t THREADS_PER_BLOCK = 1024;

    int h_x_rows = ekfom_data->h_x.rows();
    int h_x_cols = ekfom_data->h_x.cols();
    rmagine::MatrixXd h_x_raw(h_x_rows, h_x_cols);

    int h_rows = ekfom_data->h.rows();
    int h_cols = ekfom_data->h.cols();
    rmagine::MatrixXd h_raw(h_rows, h_cols);

    krnl_jacobian<<<std::ceil((float)effct_feat_num / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
        laser_cloud_ori, laser_cloud_ori_size,
        corr_normvect, corr_normvect_size,
        rmagine::Quaterniond(s->rot.w(), s->rot.x(), s->rot.y(), s->rot.z()),
        rmagine::Vector3d(s->offset_T_L_I.x(), s->offset_T_L_I.y(), s->offset_T_L_I.z()),
        rmagine::Quaterniond(s->offset_R_L_I.w(), s->offset_R_L_I.x(), s->offset_R_L_I.y(), s->offset_R_L_I.z()),
        h_x_raw.m_data, h_x_rows, h_x_cols,
        h_raw.m_data, h_rows, h_cols,
        false);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    h_x_raw.toEigenInpl(ekfom_data->h_x);
    h_raw.toEigenInpl(ekfom_data->h);
}