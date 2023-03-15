#include <gtest/gtest.h>
#include <ikd-Tree/ikd_Tree.h>

#include <cuda-utils.h>
#include <ikd-Tree/ikd_Tree_gpu_search.h>

#include <common_lib.h>
#include <math/common.h>
#include <math/math.h>

#include <iostream>

#include <use-ikfom.hpp>

#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distrib(-1.0, 1.0);

constexpr double epsilon = 1e-6;

Eigen::Quaterniond Random_Quaternion()
{
    double x = distrib(gen);
    double y = distrib(gen);
    double z = distrib(gen);
    double w = distrib(gen);
    return Eigen::Quaterniond(w, x, y, z).normalized();
}

/**
 * Testing interop with MTK structs. Rmagine structs work reliable on the GPU, while Eigen does not.
*/
TEST(rm_mtk_interop, quat) {
    auto rquat = Random_Quaternion();

    SO3 rot(rquat.w(), rquat.x(), rquat.y(), rquat.z());
    rmagine::Quaterniond rm_rot(rquat.w(), rquat.x(), rquat.y(), rquat.z());

    // conjugation
    EXPECT_NEAR(rot.conjugate().w(), rm_rot.conjugate().w, epsilon);
    EXPECT_NEAR(rot.conjugate().x(), rm_rot.conjugate().x, epsilon);
    EXPECT_NEAR(rot.conjugate().y(), rm_rot.conjugate().y, epsilon);
    EXPECT_NEAR(rot.conjugate().z(), rm_rot.conjugate().z, epsilon);

    // rotation matrix
    EXPECT_NEAR(rot.toRotationMatrix()(0, 0), rm_rot.toRotationMatrix()(0, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(0, 1), rm_rot.toRotationMatrix()(0, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(0, 2), rm_rot.toRotationMatrix()(0, 2), epsilon);

    EXPECT_NEAR(rot.toRotationMatrix()(1, 0), rm_rot.toRotationMatrix()(1, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(1, 1), rm_rot.toRotationMatrix()(1, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(1, 2), rm_rot.toRotationMatrix()(1, 2), epsilon);

    EXPECT_NEAR(rot.toRotationMatrix()(2, 0), rm_rot.toRotationMatrix()(2, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(2, 1), rm_rot.toRotationMatrix()(2, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(2, 2), rm_rot.toRotationMatrix()(2, 2), epsilon);

    // multiplication
    Eigen::Vector3d norm_vec(Random_Quaternion().x(), Random_Quaternion().y(), Random_Quaternion().z());
    Eigen::Vector3d C = rot.conjugate() * norm_vec;

    rmagine::Vector3d rm_norm_vec(norm_vec.x(), norm_vec.y(), norm_vec.z());
    EXPECT_NEAR(norm_vec.x(), rm_norm_vec.x, epsilon);
    EXPECT_NEAR(norm_vec.y(), rm_norm_vec.y, epsilon);
    EXPECT_NEAR(norm_vec.z(), rm_norm_vec.z, epsilon);

    rmagine::Vector3d rm_C = rm_rot.conjugate() * rm_norm_vec;

    EXPECT_NEAR(C.x(), rm_C.x, epsilon);
    EXPECT_NEAR(C.y(), rm_C.y, epsilon);
    EXPECT_NEAR(C.z(), rm_C.z, epsilon);

    // normalize
    Eigen::Quaterniond norm_quat = Random_Quaternion();
    norm_quat.w() += distrib(gen);
    norm_quat.x() += distrib(gen);
    norm_quat.y() += distrib(gen);
    norm_quat.z() += distrib(gen);
    rmagine::Quaterniond rm_norm_quat(norm_quat.w(), norm_quat.x(), norm_quat.y(), norm_quat.z());

    EXPECT_NEAR(norm_quat.normalized().w(), rm_norm_quat.normalized().w, epsilon);
    EXPECT_NEAR(norm_quat.normalized().x(), rm_norm_quat.normalized().x, epsilon);
    EXPECT_NEAR(norm_quat.normalized().y(), rm_norm_quat.normalized().y, epsilon);
    EXPECT_NEAR(norm_quat.normalized().z(), rm_norm_quat.normalized().z, epsilon);
    
    // reinterpret cast
    /*norm_quat = Random_Quaternion();
    rmagine::Quaterniond rm_casted = *(reinterpret_cast<rmagine::Quaterniond*>(&norm_quat));

    EXPECT_NEAR(norm_quat.w(), rm_casted.w, epsilon);
    EXPECT_NEAR(norm_quat.x(), rm_casted.x, epsilon);
    EXPECT_NEAR(norm_quat.y(), rm_casted.y, epsilon);
    EXPECT_NEAR(norm_quat.z(), rm_casted.z, epsilon);*/
}

TEST(rm_mtk_interop, vector3) {
    // reinterpret cast
    auto norm_quat = Random_Quaternion();
    vect3 t(Eigen::Vector3d(norm_quat.x(), norm_quat.y(), norm_quat.z()));
    rmagine::Vector3d rm_casted = *(reinterpret_cast<rmagine::Vector3d*>(&t));

    EXPECT_NEAR(t.x(), rm_casted.x, epsilon);
    EXPECT_NEAR(t.y(), rm_casted.y, epsilon);
    EXPECT_NEAR(t.z(), rm_casted.z, epsilon);
}