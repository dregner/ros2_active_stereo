#include "monitor_utils.hpp"
#include <X11/Xlib.h>
#include <X11/extensions/Xinerama.h>
#include <iostream>

std::vector<MonitorInfo> get_monitors() {
    std::vector<MonitorInfo> monitors;
    Display* display = XOpenDisplay(nullptr);

    if (!display) {
        std::cerr << "Erro: Não foi possível abrir o display X11." << std::endl;
        return monitors;
    }

    if (XineramaIsActive(display)) {
        int num_monitors;
        XineramaScreenInfo* screens = XineramaQueryScreens(display, &num_monitors);

        if (screens) {
            for (int i = 0; i < num_monitors; ++i) {
                monitors.push_back({
                    "Monitor_" + std::to_string(i),
                    screens[i].width,
                    screens[i].height,
                    screens[i].x_org,
                    screens[i].y_org
                });
            }
            XFree(screens);
        }
    } else {
        int screen = DefaultScreen(display);
        monitors.push_back({
            "DefaultMonitor",
            DisplayWidth(display, screen),
            DisplayHeight(display, screen),
            0,
            0
        });
    }

    XCloseDisplay(display);
    return monitors;
}