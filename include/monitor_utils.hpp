#ifndef MONITOR_UTILS_H
#define MONITOR_UTILS_h

#include <string>
#include <vector>

struct MonitorInfo {
    std::string name;
    int width;
    int height;
    int x;
    int y;
};

std::vector<MonitorInfo> get_monitors();

#endif