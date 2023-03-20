#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

class BagParser {
public:
    BagParser(std::string bag_file) : bag_file_(bag_file) {}

    void parse() {
        rosbag::Bag bag;
        try {
            bag.open(bag_file_, rosbag::bagmode::Read);
        } catch (rosbag::BagException &ex) {
            ROS_ERROR("Failed to open bag file %s: %s", bag_file_.c_str(), ex.what());
            return;
        }

        std::vector<std::string> topics;
        topics.push_back("/imu_topic");
        topics.push_back("/pointcloud_topic");

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        for (rosbag::MessageInstance const &m : view) {
            if (m.getTopic() == "/imu_topic") {
                sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                if (imu_msg != nullptr) {
                    processImuMsg(imu_msg);
                }
            } else if (m.getTopic() == "/pointcloud_topic") {
                sensor_msgs::PointCloud2ConstPtr pc_msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (pc_msg != nullptr) {
                    processPointCloud2Msg(pc_msg);
                }
            }
        }

        bag.close();
    }

private:
    void processImuMsg(const sensor_msgs::ImuConstPtr &imu_msg) {
        // Process the IMU message here
        // ...
        ROS_INFO("IMU message processed.");
    }

    void processPointCloud2Msg(const sensor_msgs::PointCloud2ConstPtr &pc_msg) {
        // Process the PointCloud2 message here
        // ...
        ROS_INFO("PointCloud2 message processed.");
    }

    std::string bag_file_;
};
