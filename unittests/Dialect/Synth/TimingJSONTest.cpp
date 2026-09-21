//===- TimingJSONTest.cpp - Timing result JSON round-trip tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "circt/Dialect/Synth/Timing/TimingResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt::synth::timing;

namespace {

struct TimingJSONTest : public ::testing::Test {
  MLIRContext ctx;
  TimingResult makeResult() {
    TimingResult result;
    result.status = TimingResultStatus::Complete;
    result.backend = StringAttr::get(&ctx, "test-backend");
    result.backendVersion = StringAttr::get(&ctx, "1");
    result.coverage.expectedEndpoints = 2;
    result.coverage.constrainedEndpoints = 2;
    result.coverage.analyzedEndpoints = 2;

    TimingObjectKey ep;
    ep.kind = TimingObjectKind::InstancePin;
    ep.instancePath.push_back(StringAttr::get(&ctx, "u_core"));
    ep.moduleName = StringAttr::get(&ctx, "DFF");
    ep.portIndex = 0;
    ep.portName = StringAttr::get(&ctx, "D");
    ep.bit = 0;

    TimingEndpointResult endpoint;
    endpoint.endpoint = ep;
    endpoint.pathGroup = StringAttr::get(&ctx, "clk");
    TimingTransitionResult rise;
    // Fill fields explicitly for clarity.
    rise.transition = TimingTransition::Rise;
    rise.arrival = TimePs{5312};
    rise.required = TimePs{5000};
    rise.slack = TimePs{-312};
    TimingObjectKey launch = ep;
    launch.portName = StringAttr::get(&ctx, "Q");
    rise.worstLaunch = launch;
    endpoint.transitions.push_back(rise);
    TimingTransitionResult fall;
    fall.transition = TimingTransition::Fall;
    fall.arrival = TimePs{5200};
    fall.required = TimePs{5000};
    fall.slack = TimePs{-200};
    endpoint.transitions.push_back(fall);
    result.endpoints.push_back(endpoint);

    TimingPathResult path;
    path.endpoint = ep;
    path.transition = TimingTransition::Rise;
    path.arrival = TimePs{5312};
    path.required = TimePs{5000};
    path.slack = TimePs{-312};
    TimingPathPoint point;
    point.object = launch;
    point.transition = TimingTransition::Rise;
    point.arrival = TimePs{100};
    path.points.push_back(point);
    result.paths.push_back(path);

    normalizeTimingResult(result);
    return result;
  }
};

TEST_F(TimingJSONTest, RoundTrip) {
  TimingResult result = makeResult();
  EXPECT_EQ(result.wns->value, -312);
  EXPECT_EQ(result.tns->value, -312);
  EXPECT_EQ(result.failingEndpoints, 1u);

  std::string json;
  llvm::raw_string_ostream os(json);
  printTimingResult(result, os);
  os.flush();

  // Schema marker present.
  EXPECT_NE(json.find("circt.synth.timing-result.v1"), std::string::npos);

  TimingResult parsed;
  EXPECT_TRUE(succeeded(parseTimingResult(json, parsed, &ctx)));
  EXPECT_EQ(parsed.status, TimingResultStatus::Complete);
  EXPECT_EQ(parsed.wns->value, -312);
  EXPECT_EQ(parsed.tns->value, -312);
  EXPECT_EQ(parsed.failingEndpoints, 1u);
  ASSERT_EQ(parsed.endpoints.size(), 1u);
  EXPECT_EQ(parsed.endpoints[0].endpoint.portName.getValue(), "D");
  ASSERT_EQ(parsed.endpoints[0].transitions.size(), 2u);
  EXPECT_EQ(parsed.endpoints[0].transitions[0].slack.value, -312);
  ASSERT_TRUE(parsed.endpoints[0].transitions[0].worstLaunch.has_value());
  EXPECT_EQ(parsed.endpoints[0].transitions[0].worstLaunch->portName.getValue(),
            "Q");
  ASSERT_EQ(parsed.paths.size(), 1u);
  ASSERT_EQ(parsed.paths[0].points.size(), 1u);
  EXPECT_EQ(parsed.paths[0].points[0].arrival.value, 100);
}

TEST_F(TimingJSONTest, RejectsBadSchema) {
  TimingResult parsed;
  EXPECT_TRUE(
      failed(parseTimingResult("{\"schema\":\"other\"}", parsed, &ctx)));
  EXPECT_TRUE(failed(parseTimingResult("not json", parsed, &ctx)));
  EXPECT_TRUE(failed(parseTimingResult("{\"schema\":"
                                       "\"circt.synth.timing-result.v1\"}",
                                       parsed, &ctx)));
}

TEST_F(TimingJSONTest, NormalizeEmpty) {
  TimingResult result;
  result.status = TimingResultStatus::Complete;
  normalizeTimingResult(result);
  EXPECT_FALSE(result.wns.has_value());
  EXPECT_FALSE(result.tns.has_value());
  EXPECT_EQ(result.failingEndpoints, 0u);
}

TEST_F(TimingJSONTest, DeterministicOrder) {
  TimingResult a = makeResult();
  TimingResult b = makeResult();
  // Reverse insertion order; output must still match.
  std::reverse(b.endpoints.begin(), b.endpoints.end());
  std::string ja, jb;
  llvm::raw_string_ostream osa(ja), osb(jb);
  printTimingResult(a, osa);
  printTimingResult(b, osb);
  osa.flush();
  osb.flush();
  EXPECT_EQ(ja, jb);
}

} // namespace
