# FindEigen.cmake — wrapper that finds Eigen3 and sets legacy variable names
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
set(EIGEN_FOUND TRUE)
