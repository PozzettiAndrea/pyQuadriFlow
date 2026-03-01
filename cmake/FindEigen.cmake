# FindEigen.cmake — wrapper that finds Eigen3 and sets legacy variable names
# needed because quadriflow uses find_package(Eigen) but brew/system provides Eigen3
find_package(Eigen3 REQUIRED)

# Eigen3 sets various variables depending on the platform/version.
# Cover all cases to get the include dir containing "Eigen/Core".
if(TARGET Eigen3::Eigen)
  get_target_property(_eigen_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  set(EIGEN_INCLUDE_DIRS "${_eigen_inc}")
elseif(EIGEN3_INCLUDE_DIR)
  set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
elseif(EIGEN3_INCLUDE_DIRS)
  set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIRS}")
endif()

set(EIGEN_FOUND TRUE)
