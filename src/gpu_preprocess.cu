#include <gpu_preprocess.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <unistd.h>
#include <pcl/point_types.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

struct EIGEN_ALIGN16 PointXYZINormal
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
};

struct EIGEN_ALIGN16 OusterPoint
{
PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointXYZ
{
    PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])

    PCL_MAKE_ALIGNED_OPERATOR_NEW
};

bool init = false;
void* input_raw_buffer; // size does not change
void* input_raw_buffer_transform;
PointXYZINormal* output_pt_buffer; // for fast_lio
PointXYZ* mapping_pt_buffer; // for TSDF Node
uint32_t* input_raw_buffer_size;

uint32_t current_size = 0;

bool work;
bool stopped;

__global__ void krnl_transform_scan(void* raw_data, uint32_t raw_data_size, uint32_t point_step, PointXYZ* output_buffer, uint32_t output_buffer_size, Eigen::Vector3f translation, Eigen::Quaternionf rotation)
{
    int32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= output_buffer_size) return;

    OusterPoint* res_ptr = (OusterPoint*) (raw_data + (point_step*id));
    PointXYZ result;
    result.x = res_ptr->x;
    result.y = res_ptr->y;
    result.z = res_ptr->z;

    Eigen::Vector3f point(result.x, result.y, result.z);
    point = rotation * point + translation;

    result.x = point.x();
    result.y = point.y();
    result.z = point.z();

    output_buffer[id] = result;
}

void ros_ouster_point_transform(unsigned char* data, uint32_t orig_size, uint32_t point_step, uint32_t raw_size, void* pcl_out, Eigen::Vector3f translation, Eigen::Quaternionf rotation)
{
    if (!work)
    {
        stopped = true;
        return;
    }

    if (orig_size == 0 || raw_size == 0 || data == nullptr)
    {
        return;
    }

    // allocate enough memory to include the current cloud
    if (current_size != 0 && raw_size > current_size)
    {
        deinit_buffer();
    }
    current_size = raw_size;

    if (input_raw_buffer_transform == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMallocManaged(&input_raw_buffer_transform, raw_size));
    }

    if (mapping_pt_buffer == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMallocManaged(&mapping_pt_buffer, sizeof(PointXYZ) * orig_size));
    }

    // copy full cloud to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(input_raw_buffer_transform, data, raw_size, cudaMemcpyHostToDevice));

    constexpr size_t THREADS_PER_BLOCK = 1024;
    size_t BLOCKS = std::ceil((float)orig_size / THREADS_PER_BLOCK);

    krnl_transform_scan<<<BLOCKS, THREADS_PER_BLOCK>>>(input_raw_buffer_transform, raw_size, point_step, mapping_pt_buffer, orig_size, translation, rotation);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(pcl_out, mapping_pt_buffer, sizeof(PointXYZ) * orig_size, cudaMemcpyDeviceToHost));
}

__global__ void krnl_filter_points_xyziring(void* raw_data, uint32_t raw_data_size, uint32_t point_step, PointXYZINormal* output_buffer, uint32_t output_buffer_size, uint32_t* out_size)
{
    int32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= output_buffer_size) return;

    //uint8_t ring = *(std::uint8_t*)((raw_data + (point_step * id)) + sizeof(PADDING4D) + sizeof(float) + sizeof(std::uint32_t) + sizeof(uint16_t));
    //if ((ring % 2) != 0) return;
    
    if ((id / 1024) % 2 != 0) return;
    if (id % 3 != 0) return;

    OusterPoint* res_ptr = (OusterPoint*) (raw_data + (point_step*id));
    PointXYZINormal result;
    result.x = res_ptr->x;
    result.y = res_ptr->y;
    result.z = res_ptr->z;

    if (result.x * result.x + result.y * result.y + result.z * result.z < 2.f * 2.f) return;

    result.intensity = res_ptr->intensity;
    result.normal_x = 0;
    result.normal_y = 0;
    result.normal_z = 0;
    result.curvature = res_ptr->t * 1.e-6f; // nanosecond with ousterpoint

    int val = atomicAdd(out_size, 1);
    output_buffer[val] = result;
}

void ros_ouster_point_to_pointcloud_xyzi(const unsigned char* data, uint32_t orig_size, uint32_t point_step, uint32_t raw_size, void* pl_orig_buffer, uint32_t* pl_orig_size)
{
    if (!work)
    {
        stopped = true;
        return;
    }

    if (orig_size == 0 || raw_size == 0 || data == nullptr)
    {
        return;
    }

    // allocate enough memory to include the current cloud
    if (current_size != 0 && raw_size > current_size)
    {
        deinit_buffer();
    }

    if (input_raw_buffer == nullptr)
    {
        std::cout << "allocig input bfufer "<< std::endl;
        CHECK_CUDA_ERROR(cudaMallocManaged(&input_raw_buffer, raw_size));
    }

    if (output_pt_buffer == nullptr)
    {
        std::cout << "allocig output bfufer "<< std::endl;
        CHECK_CUDA_ERROR(cudaMallocManaged(&output_pt_buffer, sizeof(PointXYZINormal) * orig_size));
    }

    current_size = raw_size;

    // copy full cloud to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(input_raw_buffer, data, raw_size, cudaMemcpyHostToDevice));

    constexpr size_t THREADS_PER_BLOCK = 1024;
    size_t BLOCKS = std::ceil((float)orig_size / THREADS_PER_BLOCK);

    uint32_t* out_size = nullptr;
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&out_size, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(out_size, 0, sizeof(uint32_t)));
        
    if (sizeof(OusterPoint) != point_step)
    {
        std::cout << "pointstep is not ousterpoint size. pointstep: " << point_step << " ousterpointsize: " << sizeof(OusterPoint) << std::endl;
    }

    krnl_filter_points_xyziring<<<BLOCKS, THREADS_PER_BLOCK>>>(input_raw_buffer, raw_size, point_step, output_pt_buffer, orig_size, out_size);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(pl_orig_size, out_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(pl_orig_buffer, output_pt_buffer, sizeof(OusterPoint) * (*pl_orig_size), cudaMemcpyDeviceToHost));
    if (out_size != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree(out_size));
    }

}


void deinit_buffer()
{
    if (input_raw_buffer != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree((void*)input_raw_buffer));
        input_raw_buffer = nullptr;
    }

    if (output_pt_buffer != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree((void*)output_pt_buffer));
        output_pt_buffer = nullptr;
    }

    if (input_raw_buffer_size != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree((void*)input_raw_buffer_size));
        input_raw_buffer_size = nullptr;
    }

    if (mapping_pt_buffer != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree((void*)mapping_pt_buffer));
        mapping_pt_buffer = nullptr;
    }

    if (input_raw_buffer_transform != nullptr)
    {
        CHECK_CUDA_ERROR(cudaFree((void*)input_raw_buffer_transform));
        input_raw_buffer_transform = nullptr;
    }
}

void stop_gpu_prep()
{
    work = false;
    while(!stopped)
    {
        usleep(1000);
    }

    deinit_buffer();
}

void init_kernels()
{
    work = true;
}

