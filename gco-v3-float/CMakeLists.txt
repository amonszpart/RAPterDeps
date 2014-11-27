project(gco-v3-0-float)
cmake_minimum_required(VERSION 2.8)

# BUILD
SET( dir ${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE} )
SET( EXECUTABLE_OUTPUT_PATH ${dir}/bin CACHE PATH "Build directory" FORCE)
SET( LIBRARY_OUTPUT_PATH ${dir}/lib CACHE PATH "Build directory" FORCE)

add_definitions( -DGCO_ENERGYTYPE=float)

SET( HPP_LIST
    include/gco/impl/graph.hpp
    include/gco/impl/maxflow.hpp
)

SET( H_LIST
    include/gco/block.h
    include/gco/energy.h
    include/gco/graph.h
    include/gco/LinkedBlockList.h
    include/gco/GCoptimization.h
)

INCLUDE_DIRECTORIES(
    include
)

SET( SRC_LIST
    src/example.cpp
    src/GCoptimization.cpp
    src/LinkedBlockList.cpp
)

#add_executable( ${PROJECT_NAME} ${SRC_LIST} ${H_LIST} )
ADD_LIBRARY( ${PROJECT_NAME} SHARED ${SRC_LIST} ${H_LIST} ${HPP_LIST} )
SET_TARGET_PROPERTIES( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

add_executable( test${PROJECT_NAME} ${SRC_LIST} ${H_LIST} )

