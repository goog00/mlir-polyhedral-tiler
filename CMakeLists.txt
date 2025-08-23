cmake_minimum_required(VERSION 3.20)
project(GpuTileILPSelection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === OR-Tools 根目录（按你的实际解压路径） ===
set(OR_TOOLS_ROOT "/home/sunteng/codespace/or-tools_x86_64_Ubuntu-24.04_cpp_v9.10.4067")

# !!! 必须在 find_package(ortools) 之前设置 !!!
list(APPEND CMAKE_PREFIX_PATH "${OR_TOOLS_ROOT}")
# 保险起见，直接指定 config 目录（通常在 lib/cmake/ortools）
set(ortools_DIR "${OR_TOOLS_ROOT}/lib/cmake/ortools")

# === MLIR / LLVM ===
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
if (NOT LLVM_DIR)
  set(LLVM_DIR "${MLIR_DIR}/../llvm")
endif()
find_package(LLVM REQUIRED CONFIG)

# === OR-Tools ===
find_package(ortools CONFIG REQUIRED)   # 提供 imported target: ortools::ortools
message(STATUS "Using ortools from: ${ortools_DIR}")

# 其余保持不变...
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(GpuTileILPSelection MODULE 
        # src/GpuTileILPSelectionPass.cpp
        # src/ApplyTileSizesPass.cpp 
        # src/BuildPolyRepresent.cpp
        src/BuildPolyDepend.cpp
        # src/ApplyTileSizes.cpp
      )

target_include_directories(GpuTileILPSelection PRIVATE
  ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})

target_link_libraries(GpuTileILPSelection PRIVATE
  MLIRIR MLIRSupport MLIRPass MLIRTransformUtils
  MLIRFuncDialect MLIRLinalgDialect MLIRAffineDialect
  MLIRSCFDialect MLIRGPUDialect MLIRGPUTransforms
  MLIRTensorDialect MLIRMemRefDialect MLIRArithDialect 
  ortools::ortools
  LLVMSupport
  )

mlir_check_all_link_libraries(GpuTileILPSelection)

set_target_properties(GpuTileILPSelection PROPERTIES
  PREFIX ""
  # 让插件运行时能找到 OR-Tools 的 .so（可选但推荐）
  BUILD_RPATH "${OR_TOOLS_ROOT}/lib"
  INSTALL_RPATH "${OR_TOOLS_ROOT}/lib")


message(STATUS "LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR_INCLUDE_DIRS = ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_DEFINITIONS  = ${LLVM_DEFINITIONS}")