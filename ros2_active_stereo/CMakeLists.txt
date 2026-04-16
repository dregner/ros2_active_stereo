cmake_minimum_required(VERSION 3.8)
project(ros2_active_stereo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ros2_active_stereo_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)


# Manually specify the path to gpiod
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GPIOD REQUIRED libgpiod)
# include_directories(${GPIOD_INCLUDE_DIRS})
# link_directories(${GPIOD_LIBRARY_DIRS})

# # Manually specify the path to libnoise
# set(LIBNOISE_INCLUDE_DIRS "/usr/local/include")
# set(LIBNOISE_LIBRARIES "/usr/local/lib/libnoise-static.a")
# include_directories(include ${LIBNOISE_INCLUDE_DIRS}) 

# Cria a Biblioteca Compartilhada (O Componente / Plugin)
add_library(image_project_comp SHARED
  src/image_project.cpp
  src/structured-light/FringePattern.cpp
  src/structured-light/GrayCode.cpp
  src/monitor_utils.cpp

)
target_include_directories(image_project_comp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>

)
ament_target_dependencies(image_project_comp
  rclcpp rclcpp_components std_srvs cv_bridge OpenCV 
)
# Registra a biblioteca como um plugin dinâmico do ROS 2
rclcpp_components_register_nodes(image_project_comp "ImageProjectNode")
target_link_libraries(image_project_comp
  ${X11_LIBRARIES}
  ${X11_Xinerama_LIB}
)

# Cria o Executável Standalone (Para testes com ros2 run)
add_executable(image_project_node src/image_project_node.cpp)
target_link_libraries(image_project_node image_project_comp X11 Xinerama)
ament_target_dependencies(image_project_node rclcpp)


# ==============================================================================
# FRINGE PROCESS NODE (Stereo Process)
# ==============================================================================

# Cria a Biblioteca Compartilhada (O Componente / Plugin)
add_library(stereo_process_comp SHARED
  src/stereo_process.cpp
  src/structured-light/FringeProcess.cpp
  src/structured-light/FringePattern.cpp
  src/structured-light/GrayCode.cpp
  src/monitor_utils.cpp
)

target_include_directories(stereo_process_comp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(stereo_process_comp
  rclcpp rclcpp_components sensor_msgs std_srvs message_filters cv_bridge image_transport ros2_active_stereo_msgs OpenCV
)
# Registra a biblioteca como um plugin dinâmico do ROS 2
rclcpp_components_register_nodes(stereo_process_comp "StereoProcessNode")

# Cria o Executável Standalone (Para testes com ros2 run)
add_executable(stereo_process_node src/stereo_process_node.cpp)
target_link_libraries(stereo_process_node stereo_process_comp)
ament_target_dependencies(stereo_process_node rclcpp)

# ==============================================================================
# REGRAS DE INSTALAÇÃO
# ==============================================================================

# Instala bibliotecas, executáveis e arquivos de cabeçalho
install(TARGETS
      image_project_comp
      image_project_node
      stereo_process_comp
      stereo_process_node
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib/${PROJECT_NAME}
    )

install(DIRECTORY include/
  DESTINATION include
)

install(
    PROGRAMS
    scripts/triangulation_node.py
    scripts/SpatialCorrelation_pytorch.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        launch
        config
        DESTINATION share/${PROJECT_NAME})
        

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
