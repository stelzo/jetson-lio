#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <nav_msgs/Odometry.h>

class BagParser {
public:
    BagParser(std::string bag_file, std::string imu_topic, std::string pcl_topic) : 
        bag_file_(bag_file), 
        imu_topic_(imu_topic),
        pcl_topic_(pcl_topic) {}

    void parse(
        std::function<void(const sensor_msgs::PointCloud2::ConstPtr&)> pcl_callback,
        std::function<void(const sensor_msgs::Imu::ConstPtr&)> imu_callback,
        int limit = -1
    ) {
        rosbag::Bag bag;
        try {
            bag.open(bag_file_, rosbag::bagmode::Read);
        } catch (rosbag::BagException &ex) {
            ROS_ERROR("Failed to open bag file %s: %s", bag_file_.c_str(), ex.what());
            return;
        }

        std::vector<std::string> topics;
        topics.push_back(imu_topic_);
        topics.push_back(pcl_topic_);

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        for (rosbag::MessageInstance const &m : view) {
            if (limit != -1 && limit-- == 0) break;
            if (m.getTopic() == imu_topic_) {
                sensor_msgs::ImuConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
                if (imu_msg != nullptr) {
                    imu_callback(imu_msg);
                }
            } else if (m.getTopic() == pcl_topic_) {
                sensor_msgs::PointCloud2ConstPtr pc_msg = m.instantiate<sensor_msgs::PointCloud2>();
                if (pc_msg != nullptr) {
                    pcl_callback(pc_msg);
                }
            }
        }

        bag.close();
    }

private:

    std::string bag_file_;
    std::string imu_topic_;
    std::string pcl_topic_;
};


class PointMetrics {
public:

    struct PlaneResult {
        bool found;
        Eigen::Matrix<float, 4, 1> equation;
    };

    void setMaxIdx(int max_idx) {
        this->max_idx = max_idx;
        neighbors.resize(max_idx);
        selected_surfs.resize(max_idx);
        neighbor_distances.resize(max_idx);
        planes.resize(max_idx);
        point_jacobis.resize(max_idx);
    }

    void setOdom(nav_msgs::Odometry odom) {
        this->odom = odom;
    }

    void addNeighbor(size_t idx, Eigen::Vector3d val) {
        resizeAppend(idx, val, neighbors);
    }

    void addNeighbors(int idx, const std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>>& points) {
        if (points.empty()) {
            std::cout << "No neighbors found!" << std::endl;
            return;
        }
        for (auto p : points) {
            addNeighbor(idx, {p.x, p.y, p.z});
        }
    }

    void setDistances(int idx, std::vector<float> distances) {
        setFloatArray(idx, distances, neighbor_distances);
    }

    void setFloatArray(int idx, std::vector<float> array, std::vector<std::vector<float>>& ins) {
        resizeInsert(idx, array, ins);
    }

    void setPlane(int idx, bool found, const Eigen::Matrix<float, 4, 1>& equation) {
        resizeInsert(idx, {found, equation}, planes);
    }

    void setSelectedSurf(int idx, bool selected_surf) {
        resizeInsert(idx, selected_surf, selected_surfs);
    }

    void setStateJacobi(const Eigen::MatrixXd& total) {
        state_jacobi = total;
    }

    void setPointJacobi(int idx, const std::vector<float>& point_data) {
        setFloatArray(idx, point_data, point_jacobis);
    }
    
    /// @brief 
    /// i - 1\n
    void saveSelectedSurfs(int i, boost::filesystem::path path="surfs") {
        if (neighbors.empty()) {
            std::cout << "No surfs to save!" << std::endl;
            return;
        }

        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < selected_surfs.size(); i++) {
            ofs << i << " - ";
            if (!ofs) {
                std::cout << "Could not write stream" << std::endl;
                return;
            }
            ofs << selected_surfs[i] << '\n';
        }
        ofs.close();
    }
    
    /// @brief 
    /// i - 1\n
    void loadSelectedSurfs(int i, boost::filesystem::path path="surfs") {
        neighbors.clear();
        
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        while (true) {
            int idx = -1;
            char idx_sep;
            ifs >> idx >> idx_sep;
            if (!ifs) break;
            if (idx == -1) {
                std::cout << "idx parse failed" << std::endl;
                ifs.close();
                return;
            }

            std::string seperated_ps;
            std::getline(ifs, seperated_ps);
            std::stringstream sval(seperated_ps);
            bool x;
            sval >> x;
            setSelectedSurf(idx, x);
        }
        ifs.close();
    }

    /// @brief 
    /// i - x y z, x y z....\n
    void saveNeighbors(int i, boost::filesystem::path path="neighbors") {
        if (neighbors.empty()) {
            std::cout << "No neighbors to save!" << std::endl;
            return;
        }

        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < neighbors.size(); i++) {
            if (neighbors[i].empty()) continue;
            ofs << i << " - ";
            if (!ofs) {
                std::cout << "Could not write stream" << std::endl;
                return;
            }
            std::vector<std::string> vals;
            std::transform(neighbors[i].begin(), neighbors[i].end(), std::back_inserter(vals), [](const Eigen::Vector3d& p) {
                std::stringstream ss;
                ss << p.x() << " " << p.y() << " " << p.z();
                return ss.str();
            });
            
            std::string out;
            join(vals, ',', out);
            ofs << out << '\n';
        }
        ofs.close();
    }
    
    /// @brief 
    /// i - x y z, x y z....\n
    void loadNeighbors(int i, boost::filesystem::path path="neighbors") {
        neighbors.clear();
        
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        while (true) {
            int idx = -1;
            char idx_sep;
            ifs >> idx >> idx_sep;
            if (!ifs) break;
            if (idx == -1) {
                std::cout << "idx parse failed" << std::endl;
                ifs.close();
                return;
            }

            std::string seperated_ps;
            std::getline(ifs, seperated_ps);
            std::vector<std::string> vals = split(seperated_ps, ',');

            for (auto& val : vals) {
                std::stringstream sval(val);
                double x, y, z;
                sval >> x >> y >> z;
                if (!sval) {
                    std::cout << "expecting 3 point coordinates" << std::endl;
                    ifs.close();
                    return;
                }

                addNeighbor(idx, {x, y, z});
            }
        }
        ifs.close();
    }

    /// @brief
    /// i - float, float ...\n
    void saveFloatArray(int i, boost::filesystem::path path, const std::vector<std::vector<float>>& array) {
        if (array.empty()) {
            std::cout << "Nothing to save!" << std::endl;
            return;
        }

        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < array.size(); i++) {
            ofs << i << " - ";
            if (!ofs) {
                std::cout << "Could not write stream" << std::endl;
                return;
            }
            std::vector<std::string> vals;
            std::transform(array[i].begin(), array[i].end(), std::back_inserter(vals), [](const float dist) {
                return std::to_string(dist);
            });

            std::string out;
            join(vals, ',', out);
            ofs << out << '\n';
        }
        ofs.close();
    }

    void loadFloatArray(int i, boost::filesystem::path path, std::vector<std::vector<float>>& array) {
        array.clear();

        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        while (true) {
            int idx = -1;
            char idx_sep;
            ifs >> idx >> idx_sep;
            if (!ifs) break;
            if (idx == -1) {
                std::cout << "idx parse failed" << std::endl;
                ifs.close();
                return;
            }

            std::string seperated_ps;
            std::getline(ifs, seperated_ps);
            std::vector<float> fvals;
            if (seperated_ps != " ") {
                std::vector<std::string> vals = split(seperated_ps, ',');

                for (auto& val : vals) {
                    std::stringstream sval(val);
                    float x;
                    sval >> x;
                    if (!sval) {
                        std::cout << "expecting 1 float" << std::endl;
                        ifs.close();
                        return;
                    }

                    fvals.push_back(x);
                }
            }

            setFloatArray(idx, fvals, array);
        }
        ifs.close();
    }

    
    /// @brief 
    /// i - float, float ...\n
    void saveNeighborsDistances(int i, boost::filesystem::path path="neighbor-distances") {
        saveFloatArray(i, path, neighbor_distances);
    }
 
    /// @brief 
    /// i - float,float ...\n
    void loadNeighborDistances(int i, boost::filesystem::path path="neighbor-distances") {
        loadFloatArray(i, path, neighbor_distances);
    }
    
    /// @brief 
    /// i - found x,z,y,w\n
    void savePlanes(int i, boost::filesystem::path path="planes") {
        if (planes.empty()) {
            std::cout << "No planes to save!" << std::endl;
            return;
        }

        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < neighbors.size(); i++) {
            ofs << i << " - ";
            if (!ofs) {
                std::cout << "Could not write stream" << std::endl;
                return;
            }
            
        std::vector<float> eq;
        eq.push_back(planes[i].equation(0));
        eq.push_back(planes[i].equation(1));
        eq.push_back(planes[i].equation(2));
        eq.push_back(planes[i].equation(3));
            std::vector<std::string> vals;
            std::transform(eq.begin(), eq.end(), std::back_inserter(vals), [](const float dist) {
                std::stringstream ss;
                ss << dist;
                return ss.str();
            });
            
            std::string out;
            join(vals, ',', out);
            ofs << planes[i].found << " " << out << '\n';
        }
        ofs.close();
    }
    
    void loadPlanes(int i, boost::filesystem::path path="planes") {
        planes.clear();
        
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        while (true) {
            int idx = -1;
            char idx_sep;
            ifs >> idx >> idx_sep;
            if (!ifs) break;
            if (idx == -1) {
                std::cout << "idx parse failed" << std::endl;
                ifs.close();
                return;
            }

            std::string seperated_ps;
            std::getline(ifs, seperated_ps);
            std::vector<std::string> vals = split(seperated_ps, ',');
            std::vector<float> distances;

            for (auto& val : vals) {
                std::stringstream sval(val);
                float x, y, z, w;
                bool found;
                sval >> found >> x >> y >> z >> w;
                if (!sval) {
                    std::cout << "expecting bool, x, y, z, w floats" << std::endl;
                    ifs.close();
                    return;
                }
                
                Eigen::Matrix<float, 4, 1> equation;
                equation(0) = x;
                equation(1) = y;
                equation(2) = z;
                equation(3) = w;
                
                setPlane(idx, found, equation);
            }
        }
        ifs.close();
    }

    void saveStateJacobi(int i, boost::filesystem::path path="statejacobi") {
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        for (size_t i = 0; i < state_jacobi.rows(); i++) {
            for (size_t j = 0; j < state_jacobi.cols(); j++) {
                ofs << state_jacobi(i, j) << " ";
            }
            ofs << "\n";
        }
        ofs.close();
    }
    
    void loadStateJacobi(int i, boost::filesystem::path path="neighbor-distances") {
        neighbors.clear();
        
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        size_t rows = 0;
        size_t cols = 0;

        std::vector<std::vector<float>> data;
        while (true) {
            std::string seperated_ps;
            std::getline(ifs, seperated_ps);
            if (!ifs) break;
            std::vector<std::string> vals = split(seperated_ps, ' ');
            
            data.push_back(std::vector<float>());

            for (auto& val : vals) {
                std::stringstream sval(val);
                float x;
                sval >> x;
                if (!sval) {
                    std::cout << "expecting 1 float" << std::endl;
                    ifs.close();
                    return;
                }
                data[data.size() - 1].push_back(x);
            }
        }
        ifs.close();

        Eigen::MatrixXd tmp(data.size(), data[0].size());
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                tmp(i, j) = data[i][j];
            }
        }
        
        setStateJacobi(tmp);
    }

    /// @brief
    /// i - float, float ...\n
    void savePointJacobi(int i, boost::filesystem::path path="pointjacobi") {
        saveFloatArray(i, path, point_jacobis);
    }
    
    void loadPointJacobi(int i, boost::filesystem::path path="pointjacobi") {
        loadFloatArray(i, path, point_jacobis);
    }
    
    void saveOdom(int i, boost::filesystem::path path="odom") {
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";
        std::ofstream ofs(path.c_str());
        if (!ofs) {
            std::cout << "Could not open file " << path << std::endl;
            return;
        }

        ofs << odom.pose.pose.position.x << " " << odom.pose.pose.position.y << " " << odom.pose.pose.position.z << " " << odom.pose.pose.orientation.x << " " << odom.pose.pose.orientation.y << " " << odom.pose.pose.orientation.z << " " << odom.pose.pose.orientation.w << "\n";
        ofs.close();
    }
    
    void loadOdom(int i, boost::filesystem::path path="odom") {
        boost::filesystem::create_directory(path.root_directory());
        path += "-" + std::to_string(i) + ".txt";

        std::ifstream ifs(path.c_str());
        ifs >> odom.pose.pose.position.x >> odom.pose.pose.position.y >> odom.pose.pose.position.z >> odom.pose.pose.orientation.x >> odom.pose.pose.orientation.y >> odom.pose.pose.orientation.z >> odom.pose.pose.orientation.w;
        ifs.close();
    }


    void clear() {
        neighbor_distances.clear();
        neighbors.clear();
        selected_surfs.clear();
        planes.clear();
        point_jacobis.clear();
        odom = nav_msgs::Odometry();
    }

    void assert_near_neighbors(const std::vector<std::vector<Eigen::Vector3d>>& rhs, const std::vector<std::vector<Eigen::Vector3d>>& lhs) {
        for (size_t i = 0; i < rhs.size(); i++) {
            assert(rhs[i].size() == lhs[i].size());

            for (size_t j = 0; j < rhs[i].size(); j++) {
                assert((rhs[i][j] - lhs[i][j]).norm() < 1e-3);
            }
        }
    }

    void assert_near_float_distances(const std::vector<std::vector<float>>& rhs, const std::vector<std::vector<float>>& lhs) {
        assert(rhs.size() == lhs.size());
        for (size_t i = 0; i < rhs.size(); i++) {
            assert(rhs[i].size() == lhs[i].size());

            for (size_t j = 0; j < rhs[i].size(); j++) {
                assert(std::abs(rhs[i][j] - lhs[i][j]) < 1e-3);
            }
        }
    }

    void assert_near_neighbor_distances(const std::vector<std::vector<float>>& rhs, const std::vector<std::vector<float>>& lhs) {
        assert_near_float_distances(rhs, lhs);
    }

    void assert_near_selected_surfs(const std::vector<bool>& rhs, const std::vector<bool>& lhs) {
        for (size_t i = 0; i < rhs.size(); i++) {
            assert(rhs[i] == lhs[i]);
        }
    }

    void assert_near_planes(const std::vector<PlaneResult>& rhs, const std::vector<PlaneResult>& lhs) {
        for (size_t i = 0; i < rhs.size(); i++) {
            assert(rhs[i].found == lhs[i].found);
            assert(rhs[i].equation.isApprox(lhs[i].equation, 1e-3));
        }
    }

    void assert_near_state_jacobi(const Eigen::MatrixXd& rhs, const Eigen::MatrixXd& lhs) {
        assert(rhs.rows() == lhs.rows());
        assert(rhs.cols() == lhs.cols());

        for(int i = 0; i < rhs.rows(); i++) {
            for(int j = 0; j < rhs.cols(); j++) {
                double right = rhs(i, j);
                double left = lhs(i, j);
                assert(std::abs(right - left) < 1e-3);
            }
        }
    }

    void assert_near_point_jacobis(const std::vector<std::vector<float>>& rhs, const std::vector<std::vector<float>>& lhs) {
        assert_near_float_distances(rhs, lhs);
    }

    void assert_near_odom(const nav_msgs::Odometry& rhs, const nav_msgs::Odometry& lhs) {
        assert(std::abs(rhs.pose.pose.position.x - lhs.pose.pose.position.x) < 1e-3);
        assert(std::abs(rhs.pose.pose.position.y - lhs.pose.pose.position.y) < 1e-3);
        assert(std::abs(rhs.pose.pose.position.z - lhs.pose.pose.position.z) < 1e-3);
        assert(std::abs(rhs.pose.pose.orientation.x - lhs.pose.pose.orientation.x) < 1e-3);
        assert(std::abs(rhs.pose.pose.orientation.y - lhs.pose.pose.orientation.y) < 1e-3);
        assert(std::abs(rhs.pose.pose.orientation.z - lhs.pose.pose.orientation.z) < 1e-3);
        assert(std::abs(rhs.pose.pose.orientation.w - lhs.pose.pose.orientation.w) < 1e-3);
    }

    std::vector<std::vector<Eigen::Vector3d>> neighbors;
    std::vector<std::vector<float>> neighbor_distances;
    std::vector<bool> selected_surfs;
    std::vector<PlaneResult> planes;

    Eigen::MatrixXd state_jacobi;
    std::vector<std::vector<float>> point_jacobis;

    nav_msgs::Odometry odom;

private:

    template<typename T>
    void resize(size_t idx, std::vector<T>& data)
    {
        if (data.empty() || idx >= data.size() - 1) data.resize(idx + 1);
    }

    template<typename T>
    void resizeInsert(size_t idx, T val, std::vector<T>& data) {
        resize(idx, data);
        data[idx] = val;
    }

    template<typename T>
    void resizeAppend(size_t idx, T val, std::vector<std::vector<T>>& data) {
        resize(idx, data);
        data[idx].push_back(val);
    }

    static std::vector<std::string> split(std::string str, char delimiter) { 
        std::vector<std::string> internal; 
        std::stringstream ss(str); // Turn the string into a stream. 
        std::string tok; 
        
        while(std::getline(ss, tok, delimiter)) { 
            internal.push_back(tok); 
        } 
        
        return internal; 
    } 

    static void join(const std::vector<std::string>& v, char c, std::string& s) {
        s.clear();

        for (std::vector<std::string>::const_iterator p = v.begin(); p != v.end(); ++p) {
            s += *p;
            if (p != v.end() - 1)
                s += c;
        }
    }



    int max_idx;


};
