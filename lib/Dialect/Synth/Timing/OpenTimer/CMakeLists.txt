# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Optional OpenTimer timing backend
# (docs/StaticTimingAnalysisLibraryIntegration.md). Explicit source
# manifest: timer core, Liberty model, units, and utilities. Excluded:
# Verilog/SDC/SPEF parsers, shell, tools, examples, tests.

if(NOT CIRCT_ENABLE_OPENTIMER)
  return()
endif()

find_package(Threads REQUIRED)

if("${CIRCT_OPENTIMER_SOURCE_DIR}" STREQUAL "")
  message(FATAL_ERROR
    "CIRCT_ENABLE_OPENTIMER requires CIRCT_OPENTIMER_SOURCE_DIR")
endif()
set(OT ${CIRCT_OPENTIMER_SOURCE_DIR})
if(NOT EXISTS "${OT}/ot/timer/timer.cpp")
  message(FATAL_ERROR
    "CIRCT_OPENTIMER_SOURCE_DIR lacks ot/timer/timer.cpp: ${OT}")
endif()
# The adapter needs Timer::report_endpoints() from the CIRCT patch
# (opentimer-report-endpoints.patch in this directory); fail closed on
# unpatched sources rather than erroring obscurely at build time.
file(READ "${OT}/ot/timer/timer.hpp" _ot_timer_hpp)
if(NOT _ot_timer_hpp MATCHES "report_endpoints")
  message(FATAL_ERROR
    "OpenTimer sources lack Timer::report_endpoints(). Apply "
    "lib/Dialect/Synth/Timing/OpenTimer/opentimer-report-endpoints.patch "
    "to CIRCT_OPENTIMER_SOURCE_DIR (${OT}) first.")
endif()

# Generate ot/config.hpp from the template (version strings only; the
# TCL/shell paths it mentions belong to excluded components).
set(OT_VERSION_MAJOR 2)
set(OT_VERSION_MINOR 0)
set(OT_VERSION_PATCH 0)
set(OT_VERSION "2.0.0")
set(OT_HOME "${OT}")
set(OT_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(OT_TCLSH_PATH "")
configure_file("${OT}/ot/config.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/ot-config/ot/config.hpp"
               @ONLY)

set(CIRCT_OPENTIMER_SOURCES
  ${OT}/ot/timer/arc.cpp
  ${OT}/ot/timer/celllib.cpp
  ${OT}/ot/timer/celllibs.cpp
  ${OT}/ot/timer/clock.cpp
  ${OT}/ot/timer/cppr.cpp
  ${OT}/ot/timer/dump.cpp
  ${OT}/ot/timer/endpoint.cpp
  ${OT}/ot/timer/gate.cpp
  ${OT}/ot/timer/net.cpp
  ${OT}/ot/timer/path.cpp
  ${OT}/ot/timer/pfxt.cpp
  ${OT}/ot/timer/pin.cpp
  ${OT}/ot/timer/scc.cpp
  ${OT}/ot/timer/sfxt.cpp
  ${OT}/ot/timer/test.cpp
  ${OT}/ot/timer/timer.cpp
  ${OT}/ot/timer/unit.cpp
  ${OT}/ot/liberty/cell.cpp
  ${OT}/ot/liberty/celllib.cpp
  ${OT}/ot/liberty/cellpin.cpp
  ${OT}/ot/liberty/lut.cpp
  ${OT}/ot/liberty/power.cpp
  ${OT}/ot/liberty/timing.cpp
  ${OT}/ot/unit/unit.cpp
  ${OT}/ot/utility/os.cpp
  ${OT}/ot/utility/tokenizer.cpp
)
add_library(CIRCTOpenTimer STATIC ${CIRCT_OPENTIMER_SOURCES})
# Build-only includes (the OpenTimer sources are not installed).
target_include_directories(CIRCTOpenTimer PUBLIC
  $<BUILD_INTERFACE:${OT}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/ot-config>)
target_compile_features(CIRCTOpenTimer PUBLIC cxx_std_17)
target_link_libraries(CIRCTOpenTimer PUBLIC Threads::Threads)
# Keep the third-party warning surface out of CIRCT builds.
target_compile_options(CIRCTOpenTimer PRIVATE -w)
# OpenTimer/Taskflow use exceptions and RTTI, which LLVM builds disable.
target_compile_options(CIRCTOpenTimer PRIVATE -fexceptions -frtti)
set_property(GLOBAL APPEND PROPERTY CIRCT_EXPORTS CIRCTOpenTimer)
install(TARGETS CIRCTOpenTimer EXPORT CIRCTTargets)

add_circt_dialect_library(CIRCTSynthOpenTimer
  OpenTimerTimingOracle.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Synth

  DEPENDS
  CIRCTSynthTiming

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSynthTiming
  CIRCTOpenTimer
  MLIRIR
)
# The adapter includes OpenTimer headers, so it needs the same treatment.
target_compile_options(obj.CIRCTSynthOpenTimer PRIVATE -fexceptions -frtti)
