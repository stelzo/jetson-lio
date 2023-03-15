#pragma once

#include <chrono>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

namespace jetson_lio
{

using Clock = std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;

struct TimerRecord {
    inline TimerRecord() = default;
    inline TimerRecord(const std::string& name, double time_usage) 
    {
        func_name_ = name;
        time_usage_avg_ms_ = time_usage;
        n_calls_ = 1;
    }

    inline void add(double time_usage)
    {
        n_calls_++;
        time_usage_avg_ms_ = time_usage_avg_ms_ + ((time_usage - time_usage_avg_ms_) / n_calls_);
    }

    std::string func_name_;
    size_t n_calls_;
    double time_usage_avg_ms_;
};

/// timer
class Timer {
    static inline std::map<std::string, TimerRecord>& record_map()
    {
        static std::map<std::string, TimerRecord> records_;
        return records_;
    }

   public:

    /**
     * call F and save its time usage
     * @tparam F
     * @param func
     * @param func_name
     */
    template <class F>
    static inline void Evaluate(F&& func, const std::string& func_name) 
    {
        auto t1 = Clock::now();
        std::forward<F>(func)();
        auto t2 = Clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;

        if (record_map().find(func_name) != record_map().end()) {
            record_map()[func_name].add(time_used);
        } else {
            record_map().insert({func_name, TimerRecord(func_name, time_used)});
        }
    }

    /// print the run time
    static inline void PrintAll() {
        std::cout << ">>> ===== Printing run time =====\n";
        for (const auto& r : record_map()) {
            std::cout << "> [ " << r.first << " ] average time usage: "
                      << r.second.time_usage_avg_ms_
                      << " ms , called times: " << r.second.n_calls_ << '\n';
        }
        std::cout << ">>> ===== Printing run time end =====\n";
    }

    /// dump to a log file
    static inline void DumpIntoFile(const std::string& file_name) {
        std::ofstream ofs(file_name, std::ios::out);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << file_name;
            return;
        } else {
            std::cout << "Dump Time Records into file: " << file_name;
        }

        for (const auto& iter : record_map()) {
            std::cout << "> [ " << iter.first << " ] average time usage: "
                      << iter.second.time_usage_avg_ms_
                      << " ms , called times: " << iter.second.n_calls_ << '\n';
        }
        ofs << std::endl;
        ofs.close();
    }

    /// get the average time usage of a function
    static inline double GetMeanTime(const std::string& func_name) {
        if (record_map().find(func_name) == record_map().end()) {
            return 0.0;
        }

        return record_map()[func_name].time_usage_avg_ms_;
    }


    /// clean the record_map
    static void Clear() { record_map().clear(); }

    inline Timer(const std::string& func_name)
    : func_name_(func_name)
    , stopped_(false)
    {
    }

    inline void start()
    {
        start_ = Clock::now();
    }

    inline void stop()
    {
        stop_ = Clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(stop_ - start_).count() * 1000;

        if (record_map().find(func_name_) != record_map().end()) {
            record_map()[func_name_].add(time_used);
        } else {
            record_map().insert({func_name_, TimerRecord(func_name_, time_used)});
        }

        stopped_ = true;
    }

    inline ~Timer()
    {
        if (!stopped_)
        {
            stop();
        }
    }

private:

    std::string func_name_;
    bool stopped_;
    Clock::time_point start_;
    Clock::time_point stop_;

};

} // namespace flysense