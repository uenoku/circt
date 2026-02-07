# Build CaDiCaL SAT solver as external project
include(ExternalProject)

set(CADICAL_VERSION "rel-3.0.0" CACHE STRING "CaDiCaL version/tag to use")
set(CADICAL_GIT_URL "https://github.com/arminbiere/cadical.git" CACHE STRING "CaDiCaL git repository")

# Configure CaDiCaL build location
set(CADICAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cadical)
set(CADICAL_SRC_DIR ${CADICAL_PREFIX}/src/CaDiCaL_external)
set(CADICAL_INCLUDE_DIR ${CADICAL_SRC_DIR}/src)
set(CADICAL_LIBRARY ${CADICAL_SRC_DIR}/build/libcadical.a)

# Add CaDiCaL as ExternalProject
ExternalProject_Add(CaDiCaL_external
  PREFIX ${CADICAL_PREFIX}
  GIT_REPOSITORY ${CADICAL_GIT_URL}
  GIT_TAG ${CADICAL_VERSION}
  GIT_SHALLOW ON
  UPDATE_DISCONNECTED ON
  CONFIGURE_COMMAND ./configure
  BUILD_COMMAND make -j
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${CADICAL_LIBRARY}
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
)

# Create imported library target
add_library(CaDiCaL STATIC IMPORTED GLOBAL)
set_target_properties(CaDiCaL PROPERTIES
  IMPORTED_LOCATION ${CADICAL_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${CADICAL_INCLUDE_DIR}
)
add_dependencies(CaDiCaL CaDiCaL_external)

# Export variables for use in other CMakeLists
set(CADICAL_FOUND TRUE CACHE BOOL "CaDiCaL found")
set(CADICAL_INCLUDE_DIR ${CADICAL_INCLUDE_DIR} CACHE PATH "CaDiCaL include directory")
set(CADICAL_LIBRARY ${CADICAL_LIBRARY} CACHE FILEPATH "CaDiCaL library")

message(STATUS "CaDiCaL will be built from ${CADICAL_GIT_URL} (${CADICAL_VERSION})")
