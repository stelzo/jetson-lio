//#include <ros/ros.h>
//#include <sensor_msgs/PointCloud2.h>
#include <stdint.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct mapping_point {
    float x;
    float y;
    float z;
};

typedef struct mapping_point MappingPoint;

/*
 * msg_buffer is raw pointcloud2
 * std_msgs/Header header
 *  - uint32 seq
 *  - time stamp
 *      - sec: seconds since epoch
 *      - nsec: nanoseconds since second
 *  - string frame_id
 * uint32 height
 * uint32 width
 * sensor_msgs/PointField[] fields
 * bool is_bigendian
 * uint32 point_step
 * uint32 row_step
 * uint8[] data
 * bool is_dense
 */
void ros_ouster_point_to_pointcloud_xyzi(const unsigned char* data, uint32_t orig_size, uint32_t point_step, uint32_t raw_size, void* pl_orig_buffer, uint32_t* pl_orig_size);

void deinit_buffer();

void stop_gpu_prep();
void init_kernels();

void ros_ouster_point_transform(unsigned char* data, uint32_t orig_size, uint32_t point_step, uint32_t raw_size, void* pcl_out, Eigen::Vector3f translation, Eigen::Quaternionf rotation);
