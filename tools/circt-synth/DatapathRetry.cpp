//===- DatapathRetry.cpp - STA-judged datapath retry driver --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See DatapathRetry.h. Method: baseline run, then up to N structural
// variants (datapath lowering axes only; mapping policy stays as set).
// Each candidate is judged by OpenSTA worst slack (then instance count)
// and the best is kept, so retry can never regress the baseline.
//
//===----------------------------------------------------------------------===//

#include "DatapathRetry.h"
#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "circt/Dialect/Synth/Timing/TimingResult.h"
#include "circt/Dialect/Synth/Timing/TimingTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#ifdef CIRCT_OPENTIMER_ENABLED
#include "circt/Dialect/Synth/Timing/TimingDesignBuilder.h"
#include "circt/Dialect/Synth/Timing/TimingOracle.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#endif


#include <cmath>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace circt::synth;
using namespace llvm;
using namespace mlir;

namespace {

// Structural variants in a fixed heuristic order: (name, extra flags).
const std::vector<std::pair<std::string, std::vector<std::string>>> kVariants =
    {
        {"force-booth", {"--force-booth"}},
        {"compress-to-add", {"--compress-to-add"}},
        {"booth+compress-to-add", {"--force-booth", "--compress-to-add"}},
        {"timing-unaware", {"--disable-timing-aware"}},
        {"no-datapath", {"--disable-datapath"}},
};

// Shell-quote one argv token.
std::string shellQuote(StringRef token) {
  if (token.empty())
    return "''";
  if (token.find_first_of(" \t\n\"'\\$`!") == StringRef::npos)
    return token.str();
  std::string quoted = "'";
  for (char c : token) {
    if (c == '\'')
      quoted += "'\\''";
    else
      quoted += c;
  }
  quoted += "'";
  return quoted;
}

std::string joinFlags(ArrayRef<std::string> flags) {
  std::string out;
  for (const auto &flag : flags) {
    if (!out.empty())
      out += " ";
    out += flag;
  }
  return out;
}

// Consume a `--flag value` or `--flag=value` token at args[i]; returns
// tokens consumed (0 when the head is not the named flag). Value-taking
// flags consume two tokens unless joined with `=`; boolean flags consume
// exactly one (never eat the following token!).
unsigned consumeFlag(const std::vector<std::string> &args, size_t i,
                     StringRef name, bool takesValue = true) {
  if (i >= args.size())
    return 0;
  StringRef head = args[i];
  if (head == name)
    return takesValue ? (i + 1 < args.size() ? 2 : 1) : 1;
  if (head.starts_with((name + "=").str()))
    return 1;
  return 0;
}

struct AlarmState {
  static volatile sig_atomic_t fired;
  static void handler(int) { fired = 1; }
};
volatile sig_atomic_t AlarmState::fired = 0;

int runCommand(const std::string &command, unsigned timeoutSecs) {
  // Fork+exec (never system()): each candidate gets its own process group
  // so a timeout SIGKILLs the whole subtree (synth grandchildren included).
  // No external `timeout` dependency; hermetic in any test environment.
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    setpgid(0, 0);
    execl("/bin/sh", "sh", "-c", command.c_str(), (char *)nullptr);
    _exit(127);
  }
  // Best-effort group join (child also joins; closes the exec race).
  setpgid(pid, pid);
  struct sigaction sa {
  }, oldsa{};
  sa.sa_handler = AlarmState::handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGALRM, &sa, &oldsa);
  AlarmState::fired = 0;
  if (timeoutSecs > 0)
    alarm(timeoutSecs);
  int status = 0;
  int rc = -1;
  while (true) {
    pid_t w = waitpid(pid, &status, 0);
    if (w == pid) {
      rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
      break;
    }
    if (errno == EINTR && AlarmState::fired) {
      kill(-pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      rc = 124; // timeout convention
      break;
    }
    if (errno != EINTR) {
      rc = -1;
      break;
    }
  }
  alarm(0);
  sigaction(SIGALRM, &oldsa, nullptr);
  return rc;
}

std::string siblingTool(StringRef selfPath, StringRef tool) {
  StringRef dir = sys::path::parent_path(selfPath);
  if (dir.empty())
    return tool.str(); // Rely on PATH.
  SmallString<256> path(dir);
  sys::path::append(path, tool);
  return std::string(path.str());
}

// One STA path record from text parsing (adapter input, not scoring).
struct TextPath {
  std::string startpoint;
  std::string endpoint;
  std::string pathGroup;
  double slackNs = 0.0;
  bool hasSlack = false;
  double arrivalNs = 0.0;
};

// v1 feedback record (display names, nanoseconds). Patch 2 replaces these
// with resolved v2 keys; toStaPaths derives them from TimingResult.
struct StaPath {
  std::string startpoint;
  std::string endpoint;
  std::string pathGroup;
  double slackNs = 0.0;
  bool hasSlack = false;
  double arrivalNs = 0.0;
};

struct CandidateResult {
  std::string name;
  std::string flags;
  bool ok = false;
  int64_t instances = -1;
  // Worst arrival in picoseconds (-1 when unknown); display and the
  // no-slack comparison only. Scoring uses timing below.
  int64_t arrivalPs = -1;
  // Structured timing (populated by the text adapter today, backends
  // tomorrow). WNS/TNS/failing-endpoints normalized per endpoint.
  timing::TimingResult timing;
};

// Parent side of the timing backend: fork a --timing-run-only child with
// timeout, read its result JSON, and convert to a TimingResult. Any failure
// (missing backend, child crash/timeout, bad JSON) returns failure so the
// caller falls back to STA; the baseline is never lost to the backend.
std::optional<timing::TimingResult>
runOpenTimerChild(const DatapathRetryConfig &config,
                  mlir::MLIRContext *ctx, StringRef mlirFile,
                  StringRef base) {
  using namespace timing;
  if (config.timingBackend != "opentimer")
    return std::nullopt;
  std::string jsonFile = (Twine(base) + ".timing.json").str();
  std::string cmd = shellQuote(config.selfPath) + " --timing-run-only " +
                    shellQuote(mlirFile) + " -o " + shellQuote(jsonFile);
  // Forward backend configuration explicitly (children drop retry flags).
  cmd += " --timing-backend=opentimer";
  cmd += " --timing-clock-period-ns=" + std::to_string(config.timingClockPeriodNs);
  cmd += " --timing-clock-port=" + shellQuote(config.timingClockPort);
  cmd += " --timing-input-slew-ps=" + std::to_string(config.timingInputSlewPs);
  cmd += " --top " + shellQuote(config.top);
  cmd += " --liberty " + shellQuote(config.liberty);
  if (runCommand(cmd, config.timeoutSecs) != 0 || !sys::fs::exists(jsonFile))
    return std::nullopt;
  auto buffer = MemoryBuffer::getFile(jsonFile);
  if (!buffer)
    return std::nullopt;
  TimingResult result;
  if (failed(parseTimingResult(buffer.get()->getBuffer(), result, ctx)))
    return std::nullopt;
  return result;
}

// Shared candidate runner: synthesis, export, STA, and counting in
// isolated subprocesses with timeouts (see DatapathRetry.h).
struct RetryRunner {
  DatapathRetryConfig config;
  std::string tmpDir;
  std::string circtOpt;
  mlir::MLIRContext *mlirContext = nullptr;
  bool runCandidate(StringRef name, ArrayRef<std::string> variantFlags,
                    CandidateResult &result);
};

std::optional<double> parseTrailingNumber(StringRef line, StringRef tag) {
  // Find `<number> <tag>` at end of line; manual scan (no regex
  // dialect subtleties). Returns nullopt when absent/unparsable.
  size_t pos = line.find(tag);
  if (pos == StringRef::npos)
    return std::nullopt;
  StringRef head = line.substr(0, pos).rtrim();
  size_t end = head.size();
  size_t begin = end;
  while (begin > 0 && (isDigit(head[begin - 1]) || head[begin - 1] == '.' ||
                       head[begin - 1] == '-' || head[begin - 1] == '+' ||
                       head[begin - 1] == 'e' || head[begin - 1] == 'E'))
    --begin;
  if (begin == end)
    return std::nullopt;
  double value = 0.0;
  // NB: getAsDouble returns true on failure.
  if (head.substr(begin, end - begin).getAsDouble(value))
    return std::nullopt;
  return value;
}

bool RetryRunner::runCandidate(StringRef name,
                               ArrayRef<std::string> variantFlags,
                               CandidateResult &result) {
  result.name = name.str();
  result.flags = joinFlags(variantFlags);
  std::string base = (Twine(tmpDir) + "/" + name).str();
  std::string mlirFile = base + ".mlir";
  std::string sdcFile = base + ".sdc";
  std::string verilogFile = base + ".v";

  // Rebuild argv: drop retry control + output plumbing, add variant.
  // argv[0] is the tool itself; it is re-added by the caller below.
  std::vector<std::string> args(config.argv.begin() + 1, config.argv.end());
  std::string command;
  for (size_t i = 0; i < args.size();) {
    unsigned n = 0;
    if ((n = consumeFlag(args, i, "--datapath-retry")) ||
        (n = consumeFlag(args, i, "--retry-timeout")) ||
        (n = consumeFlag(args, i, "--retry-slack-target")) ||
        (n = consumeFlag(args, i, "--retry-min-improvement-ps")) ||
        (n = consumeFlag(args, i, "--datapath-refinement",
                         /*takesValue=*/false)) ||
        (n = consumeFlag(args, i, "--datapath-refinement-iterations")) ||
        (n = consumeFlag(args, i, "--datapath-pressure-margin-levels")) ||
        (n = consumeFlag(args, i, "--datapath-refinement-policy")) ||
        (n = consumeFlag(args, i, "--datapath-policy-file")) ||
        (n = consumeFlag(args, i, "--timing-feedback")) ||
        (n = consumeFlag(args, i, "--timing-feedback-report")) ||
        (n = consumeFlag(args, i, "--datapath-refinement-report")) ||
        (n = consumeFlag(args, i, "--retry-keep-temps",
                         /*takesValue=*/false)) ||
        (n = consumeFlag(args, i, "--sdc-output"))) {
      i += n;
      continue;
    }
    if (args[i] == "-o") {
      i += (i + 1 < args.size() ? 2 : 1);
      continue;
    }
    if (StringRef(args[i]).starts_with("-o") && args[i].size() > 2) {
      ++i; // Joined -o<file> form.
      continue;
    }
    command += shellQuote(args[i]) + " ";
    ++i;
  }
  // Force single-run children (no recursion) with variant flags.
  command += "--datapath-retry=0 ";
  for (const auto &flag : variantFlags)
    command += shellQuote(flag) + " ";
  command += "--sdc-output " + shellQuote(sdcFile) + " ";
  command += "-o " + shellQuote(mlirFile);
  std::string synthCmd = shellQuote(config.selfPath) + " " + command;
  if (runCommand(synthCmd, config.timeoutSecs) != 0 ||
      !sys::fs::exists(mlirFile)) {
    errs() << "circt-synth: retry candidate '" << name
           << "' synth failed (skipped)\n";
    return false;
  }

  // In-process timing backend first (no Verilog export or sta needed).
  // Success skips export/STA entirely; any failure falls through to STA.
  if (config.timingBackend == "opentimer") {
    bool backendOk = false;
    if (auto backendResult =
            runOpenTimerChild(config, mlirContext, mlirFile, base)) {
      result.timing = std::move(*backendResult);
      // Worst arrival for display and the no-slack comparison.
      int64_t worstArrival = -1;
      for (const auto &endpoint : result.timing.endpoints) {
        for (const auto &tr : endpoint.transitions) {
          if (tr.arrival)
            worstArrival = std::max(worstArrival, tr.arrival->value);
        }
      }
      result.arrivalPs = worstArrival;
      // Area via the authoritative resource analysis (same as STA path).
      std::string countJson = base + ".count.json";
      std::string countCmd =
          shellQuote(circtOpt) + " " +
          shellQuote("--synth-print-resource-usage-analysis=top-module-name=" +
                     config.top + " output-file=" + countJson +
                     " emit-json=true") +
          " " + shellQuote(mlirFile) + " -o /dev/null";
      if (runCommand(countCmd, config.timeoutSecs) == 0 &&
          sys::fs::exists(countJson)) {
        if (auto countBuffer = MemoryBuffer::getFile(countJson)) {
          if (auto parsed = json::parse(countBuffer.get()->getBuffer())) {
            if (const auto *arr = parsed->getAsArray(); arr && !arr->empty()) {
              if (const auto *obj = (*arr)[0].getAsObject()) {
                if (const auto *total = obj->get("total")) {
                  if (const auto *totalObj = total->getAsObject()) {
                    if (auto inst = totalObj->getNumber("hw.instance"))
                      result.instances = (int64_t)*inst;
                  }
                }
              }
            }
          }
        }
      }
      if (result.instances >= 0) {
        result.ok = true;
        if (!config.timingResult.empty()) {
          SmallString<256> tmp(config.timingResult);
          tmp += ".tmp";
          std::error_code ec;
          raw_fd_ostream out(tmp, ec);
          if (!ec) {
            printTimingResult(result.timing, out);
            out.close();
            if (!out.has_error())
              sys::fs::rename(tmp, config.timingResult);
          }
        }
        backendOk = true;
      }
    }
    if (backendOk)
      return true;
    errs() << "circt-synth: retry candidate '" << name
           << "' timing backend failed (falling back to STA)\n";
  }

  // Export (disallowReplicate keeps STA parsers happy; see item 1).
  std::string exportCmd =
      shellQuote(circtOpt) +
      " --test-apply-lowering-options=options=disallowReplicate"
      " --export-verilog " +
      shellQuote(mlirFile) + " -o /dev/null > " + shellQuote(verilogFile);
  if (runCommand(exportCmd, config.timeoutSecs) != 0 ||
      !sys::fs::exists(verilogFile)) {
    errs() << "circt-synth: retry candidate '" << name
           << "' export failed (skipped)\n";
    return false;
  }

  // STA through the candidate SDC (clocks come from the file itself).
  // Clockless designs (no create_clock in the SDC) use an
  // unconstrained max-delay report and judge by arrival.
  bool hasClocks = false;
  if (auto sdcBuffer = MemoryBuffer::getFile(sdcFile)) {
    StringRef sdcText = sdcBuffer.get()->getBuffer();
    hasClocks = sdcText.contains("create_clock");
  }
  std::string tclFile = base + ".tcl";
  {
    std::string tcl;
    SmallVector<StringRef> libs;
    StringRef(config.liberty).split(libs, ',');
    for (StringRef lib : libs) {
      lib = lib.trim();
      if (!lib.empty())
        tcl += (std::string("read_liberty ") + lib.str() + "\n");
    }
    tcl += (std::string("read_verilog ") + verilogFile + "\n");
    tcl += (std::string("link_design ") + config.top + "\n");
    tcl += (std::string("read_sdc ") + sdcFile + "\n");
    tcl += hasClocks ? "report_checks -path_delay max -digits 6\n"
                     : "report_checks -path_delay max -unconstrained "
                       "-digits 6\n";
    std::error_code ec;
    raw_fd_ostream tclOut(tclFile, ec);
    if (ec) {
      errs() << "circt-synth: cannot write STA script\n";
      return false;
    }
    tclOut << tcl;
  }
  std::string staLog = base + ".sta.log";
  std::string staCmd = "sta -no_init -exit " + shellQuote(tclFile) + " > " +
                       shellQuote(staLog) + " 2>&1";
  if (runCommand(staCmd, config.timeoutSecs) != 0 ||
      !sys::fs::exists(staLog)) {
    errs() << "circt-synth: retry candidate '" << name
           << "' STA failed (needs `sta` on PATH; skipped)\n";
    return false;
  }
  auto buffer = MemoryBuffer::getFile(staLog);
  if (!buffer) {
    errs() << "circt-synth: cannot read STA log\n";
    return false;
  }
  std::string log = buffer.get()->getBuffer().str();
  // All slack/arrival lines (report_checks lists worst first, but
  // min/max over all lines matches scoreboard semantics robustly).
  // Per-path records are grouped by endpoint into the structured result;
  // group minima make TNS per-endpoint-once instead of per-path.
  // Field order varies (arrival before slack in real report_checks, the
  // reverse in some scoreboards), so a path completes when both fields
  // are known, at the next path, or at end of log. First field occurrence
  // wins within a path.
  std::optional<double> arrival;
  std::vector<TextPath> textPaths;
  TextPath curPath;
  bool inPath = false;
  bool curHasArrival = false;
  auto pushPath = [&] {
    if (inPath && curHasArrival) {
      textPaths.push_back(curPath);
      inPath = false;
    }
  };
  auto stripPrefix = [](StringRef line,
                        StringRef prefix) -> std::optional<std::string> {
    if (!line.starts_with(prefix))
      return std::nullopt;
    return line.substr(prefix.size()).trim().str();
  };
  SmallVector<StringRef> lines;
  StringRef(log).split(lines, '\n');
  for (StringRef line : lines) {
    StringRef trimmed = line.trim();
    if (auto start = stripPrefix(trimmed, "Startpoint:")) {
      pushPath();
      curPath = TextPath();
      curHasArrival = false;
      curPath.startpoint = *start;
      inPath = true;
      continue;
    }
    if (auto end = stripPrefix(trimmed, "Endpoint:")) {
      if (inPath)
        curPath.endpoint = *end;
      continue;
    }
    if (auto group = stripPrefix(trimmed, "Path Group:")) {
      if (inPath)
        curPath.pathGroup = *group;
      continue;
    }
    if (auto s = parseTrailingNumber(line, "slack")) {
      if (inPath && !curPath.hasSlack) {
        curPath.slackNs = *s;
        curPath.hasSlack = true;
        if (curHasArrival)
          pushPath();
      }
      continue;
    }
    if (auto a = parseTrailingNumber(line, "data arrival time")) {
      arrival = !arrival ? a : std::max(*arrival, *a);
      if (inPath && !curHasArrival) {
        curPath.arrivalNs = *a;
        curHasArrival = true;
        if (curPath.hasSlack)
          pushPath();
      }
      continue;
    }
  }
  pushPath();
  if (!arrival) {
    errs() << "circt-synth: retry candidate '" << name
           << "' STA unparsable (skipped)\n";
    return false;
  }
  result.arrivalPs = llround(*arrival * 1000.0);
  // Text adapter: group records by endpoint display string into one
  // structured endpoint each (import-only identity; v2 feedback resolves
  // real objects). Each record becomes one rise transition with
  // required = arrival - slack. Unknown transition direction defaults
  // to rise; the minimum over transitions still normalizes correctly.
  // The report is treated as complete (as the judge always has); backend
  // adapters must establish coverage honestly instead.
  using namespace timing;
  mlir::MLIRContext &ctx = *mlirContext;
  TimingResult timing;
  timing.status = TimingResultStatus::Complete;
  timing.backend = mlir::StringAttr::get(&ctx, "external-opensta");
  timing.backendVersion = mlir::StringAttr::get(&ctx, "text-report");
  std::map<std::string, size_t> endpointIndex;
  for (const auto &path : textPaths) {
    auto it = endpointIndex.find(path.endpoint);
    size_t idx;
    if (it == endpointIndex.end()) {
      idx = timing.endpoints.size();
      endpointIndex[path.endpoint] = idx;
      TimingEndpointResult endpoint;
      endpoint.endpoint.kind = TimingObjectKind::InstancePin;
      endpoint.endpoint.moduleName = mlir::StringAttr::get(&ctx, "");
      endpoint.endpoint.portName = mlir::StringAttr::get(&ctx, path.endpoint);
      endpoint.pathGroup = mlir::StringAttr::get(&ctx, path.pathGroup);
      timing.endpoints.push_back(std::move(endpoint));
    } else {
      idx = it->second;
    }
    TimingTransitionResult tr;
    tr.transition = TimingTransition::Rise;
    TimePs arrival{llround(path.arrivalNs * 1000.0)};
    TimePs slack{llround((path.hasSlack ? path.slackNs : 0.0) * 1000.0)};
    tr.arrival = arrival;
    tr.slack = slack;
    tr.required = TimePs{arrival.value - (path.hasSlack ? slack.value : 0)};
    TimingObjectKey launch;
    launch.kind = TimingObjectKind::InstancePin;
    launch.moduleName = mlir::StringAttr::get(&ctx, "");
    launch.portName = mlir::StringAttr::get(&ctx, path.startpoint);
    tr.worstLaunch = launch;
    timing.endpoints[idx].transitions.push_back(std::move(tr));
    TimingPathResult pathResult;
    pathResult.endpoint = timing.endpoints[idx].endpoint;
    pathResult.transition = TimingTransition::Rise;
    pathResult.arrival = arrival;
    pathResult.required = tr.required.value_or(TimePs{0});
    pathResult.slack = slack;
    timing.paths.push_back(std::move(pathResult));
  }
  timing.coverage.expectedEndpoints = timing.endpoints.size();
  timing.coverage.constrainedEndpoints = timing.endpoints.size();
  timing.coverage.analyzedEndpoints = timing.endpoints.size();
  normalizeTimingResult(timing);
  result.timing = std::move(timing);

  // Area via the authoritative resource analysis. The pass options
  // form a SINGLE argv token (spaces split internally by the pass
  // parser); shellQuote it whole.
  std::string countJson = base + ".count.json";
  std::string countCmd =
      shellQuote(circtOpt) + " " +
      shellQuote("--synth-print-resource-usage-analysis=top-module-name=" +
                 config.top + " output-file=" + countJson +
                  " emit-json=true") +
      " " + shellQuote(mlirFile) + " -o /dev/null";
  if (runCommand(countCmd, config.timeoutSecs) != 0 ||
      !sys::fs::exists(countJson)) {
    errs() << "circt-synth: retry candidate '" << name
           << "' counting failed (skipped)\n";
    return false;
  }
  auto countBuffer = MemoryBuffer::getFile(countJson);
  int64_t instances = -1;
  if (countBuffer) {
    if (auto parsed = json::parse(countBuffer.get()->getBuffer())) {
      if (const auto *arr = parsed->getAsArray(); arr && !arr->empty()) {
        if (const auto *obj = (*arr)[0].getAsObject()) {
          if (const auto *total = obj->get("total")) {
            if (const auto *totalObj = total->getAsObject()) {
              if (auto inst = totalObj->getNumber("hw.instance"))
                instances = (int64_t)*inst;
            }
          }
        }
      }
    }
  }
  if (instances < 0) {
    errs() << "circt-synth: retry candidate '" << name
           << "' count missing instances (skipped)\n";
    return false;
  }
  result.instances = instances;
  result.ok = true;
  return true;
}

} // namespace

// Run the in-process timing backend once over a synthesized MLIR file and
// write the structured result JSON atomically. Used by the --timing-run-only
// child (isolated by fork+exec with timeout, like other candidates). Returns
// true on success. Without CIRCT_OPENTIMER_ENABLED always fails.
bool circt::synth::runTimingBackendOnce(const DatapathRetryConfig &config,
                          mlir::MLIRContext *ctx, StringRef mlirPath,
                          StringRef jsonPath) {
#ifdef CIRCT_OPENTIMER_ENABLED
  using namespace timing;
  if (!ctx || config.timingClockPeriodNs <= 0.0)
    return false;
  auto buffer = MemoryBuffer::getFile(mlirPath);
  if (!buffer)
    return false;
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(buffer.get()->getBuffer(), ctx);
  if (!module)
    return false;
  auto top = dyn_cast<circt::hw::HWModuleOp>(module->lookupSymbol(config.top));
  if (!top)
    return false;
  // Library cells are extern modules plus defined modules carrying
  // Liberty data (mapped technology cells embed synth.liberty.cell).
  // Supergates have mapping_cost but no Liberty backing, so they recurse
  // (timed via primitives) instead of binding as leaves.
  llvm::DenseSet<StringAttr> libraryCells;
  for (auto op : module->getOps<circt::hw::HWModuleExternOp>())
    libraryCells.insert(op.getModuleNameAttr());
  for (auto op : module->getOps<circt::hw::HWModuleOp>()) {
    if (op->hasAttr("synth.liberty.cell"))
      libraryCells.insert(op.getModuleNameAttr());
  }
  if (libraryCells.empty())
    return false;
  // Sort into dominance order first (mappers emit use-before-def for
  // feedback loops; the builder assumes dominance for wiring ops).
  {
    PassManager pm(ctx);
    pm.addPass(synth::createSortTopologically());
    if (failed(pm.run(*module)))
      return false;
    top = dyn_cast<hw::HWModuleOp>(module->lookupSymbol(config.top));
    if (!top)
      return false;
  }
  TimingDesignBuilder builder(ctx, top.getModuleNameAttr(), libraryCells);
  auto design = builder.build(*module);
  if (failed(design)) {
    for (const auto &diag : builder.getDiagnostics().unsupported)
      llvm::errs() << "circt-synth: timing graph: "
                   << diag.message.getValue() << "\n";
    return false;
  }
  // Constraints: single clock from a top-level input port plus default
  // input boundary conditions. Outputs stay unconstrained in v1.
  TimingRequest request;
  request.design = &*design;
  SmallVector<StringRef> libs;
  StringRef(config.liberty).split(libs, ',');
  for (StringRef lib : libs) {
    lib = lib.trim();
    if (!lib.empty())
      request.libertyFiles.push_back(lib.str());
  }
  if (request.libertyFiles.empty())
    return false;
  TimingClock clock;
  clock.name = StringAttr::get(ctx, config.timingClockPort);
  bool clockFound = false;
  for (const auto &port : top.getPortList()) {
    if (port.isInput() && port.name.getValue() == config.timingClockPort) {
      clock.source.kind = TimingObjectKind::TopPort;
      clock.source.portName = port.name;
      clock.source.portIndex = (unsigned)port.argNum;
      clock.source.bit = 0;
      clockFound = true;
      break;
    }
  }
  if (!clockFound)
    return false;
  clock.period = TimePs{llround(config.timingClockPeriodNs * 1000.0)};
  request.constraints.clocks.push_back(clock);
  for (const auto &port : top.getPortList()) {
    if (!port.isInput())
      continue;
    auto width = port.type.getIntOrFloatBitWidth();
    for (int i = 0; i < width; ++i) {
      TimingPortConstraint input;
      input.port.kind = TimingObjectKind::TopPort;
      input.port.portName = port.name;
      input.port.portIndex = (unsigned)port.argNum;
      input.port.bit = (unsigned)i;
      input.lateArrival = TimePs{0};
      input.riseSlew = TimePs{llround(config.timingInputSlewPs)};
      input.fallSlew = TimePs{llround(config.timingInputSlewPs)};
      request.constraints.ports.push_back(input);
    }
  }
  auto oracle = createOpenTimerTimingOracle(ctx);
  auto result = oracle->analyze(request);
  if (failed(result))
    return false;
  std::string json;
  { llvm::raw_string_ostream os(json);
    printTimingResult(*result, os); }
  // Atomic write: temp file plus rename.
  SmallString<256> tmp(jsonPath);
  tmp += ".tmp";
  std::error_code ec;
  raw_fd_ostream out(tmp, ec);
  if (ec)
    return false;
  out << json;
  out.close();
  if (out.has_error() || sys::fs::rename(tmp, jsonPath))
    return false;
  return true;
#else
  (void)config;
  (void)ctx;
  (void)mlirPath;
  (void)jsonPath;
  return false;
#endif
}

// Emit the winning candidate's MLIR (and SDC when requested).
bool emitWinner(const RetryRunner &runner, const CandidateResult &best) {
  const DatapathRetryConfig &config = runner.config;
  std::string winnerMlir = runner.tmpDir + "/" + best.name + ".mlir";
  if (config.outputFile == "-") {
    auto out = MemoryBuffer::getFile(winnerMlir);
    if (!out) {
      errs() << "circt-synth: cannot read winner\n";
      return false;
    }
    outs() << out.get()->getBuffer();
  } else if (sys::fs::copy_file(winnerMlir, config.outputFile)) {
    errs() << "circt-synth: cannot write " << config.outputFile << "\n";
    return false;
  }
  if (!config.userSdcOutput.empty()) {
    std::string winnerSdc = runner.tmpDir + "/" + best.name + ".sdc";
    if (sys::fs::copy_file(winnerSdc, config.userSdcOutput))
      errs() << "circt-synth: warning: cannot write " << config.userSdcOutput
             << "\n";
  }
  return true;
}

// Refinement comparator: target met dominates (min area among meeting),
// else max WNS, then max TNS, then fewest violators, then min area.
// All timing compares on integral picoseconds. Strict inequalities only,
// so the baseline wins all ties.
bool isBetterRefinement(const CandidateResult &a, const CandidateResult &b,
                        double slackTarget) {
  int64_t targetPs = llround(slackTarget * 1000.0);
  auto getWns = [](const CandidateResult &r) -> std::optional<int64_t> {
    if (r.timing.wns)
      return r.timing.wns->value;
    return std::nullopt;
  };
  auto aWns = getWns(a);
  auto bWns = getWns(b);
  bool aSat = aWns && *aWns >= targetPs;
  bool bSat = bWns && *bWns >= targetPs;
  if (aSat != bSat)
    return aSat;
  if (aSat)
    return a.instances < b.instances;
  if (aWns && bWns) {
    if (*aWns != *bWns)
      return *aWns > *bWns;
    int64_t aTns = a.timing.tns ? a.timing.tns->value : 0;
    int64_t bTns = b.timing.tns ? b.timing.tns->value : 0;
    if (aTns != bTns)
      return aTns > bTns;
    if (a.timing.failingEndpoints != b.timing.failingEndpoints)
      return a.timing.failingEndpoints < b.timing.failingEndpoints;
    return a.instances < b.instances;
  }
  if (!aWns && !bWns) {
    if (a.arrivalPs != b.arrivalPs)
      return a.arrivalPs < b.arrivalPs;
    return a.instances < b.instances;
  }
  return (bool)aWns;
}

void printRefinementTable(ArrayRef<CandidateResult> results, size_t bestIdx,
                          double slackTarget) {
  errs() << "circt-synth: refinement results (slack target " << slackTarget
         << "ns):\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    errs() << "  [" << (i == bestIdx ? "BEST" : "    ") << "] " << r.name;
    if (!r.flags.empty())
      errs() << " (" << r.flags << ")";
    errs() << ": inst=" << r.instances;
    if (r.timing.wns)
      errs() << " slack=" << (r.timing.wns->value / 1000.0) << "ns tns="
             << (r.timing.tns ? r.timing.tns->value / 1000.0 : 0.0)
             << "ns viol=" << r.timing.failingEndpoints;
    errs() << " arrival=" << (r.arrivalPs / 1000.0) << "ns\n";
  }
}

std::string sanitizeName(StringRef name) {
  std::string out;
  for (char c : name) {
    if (isAlnum(c))
      out += c;
    else
      out += '_';
  }
  if (out.empty())
    out = "region";
  return out;
}

// One refinement region from a decision report.
struct ReportRegion {
  std::string id;
  bool violated = false;
  std::optional<int64_t> worstSlackLevels;
  std::vector<std::string> endpoints;
};

// Minimal decision-report reader: regions with id/violated/slack/endpoints.
// Empty on any failure (driver proceeds with no candidates).
std::vector<ReportRegion> readRefinementReport(const std::string &path) {
  std::vector<ReportRegion> regions;
  auto file = MemoryBuffer::getFile(path);
  if (!file)
    return regions;
  auto parsed = json::parse(file.get()->getBuffer());
  if (!parsed)
    return regions;
  const auto *obj = parsed->getAsObject();
  if (!obj)
    return regions;
  const auto *list = obj->getArray("regions");
  if (!list)
    return regions;
  for (const auto &entry : *list) {
    const auto *regionObj = entry.getAsObject();
    if (!regionObj)
      continue;
    auto id = regionObj->getString("id");
    if (!id)
      continue;
    ReportRegion region;
    region.id = id->str();
    if (auto violated = regionObj->getBoolean("violated"))
      region.violated = *violated;
    if (auto slack = regionObj->getInteger("worst_slack_levels"))
      region.worstSlackLevels = *slack;
    if (const auto *endpoints = regionObj->getArray("endpoints")) {
      for (const auto &endpoint : *endpoints)
        if (auto text = endpoint.getAsString())
          region.endpoints.push_back(text->str());
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

// v1 feedback view over a structured result: one record per transition
// with display names and nanosecond values. Patch 2 replaces this with
// resolved v2 keys; until then the text adapter's import-only identity
// flows through unchanged.
std::vector<StaPath> toStaPaths(const timing::TimingResult &timing) {
  std::vector<StaPath> paths;
  for (const auto &endpoint : timing.endpoints) {
    for (const auto &tr : endpoint.transitions) {
      StaPath record;
      record.endpoint = endpoint.endpoint.portName.str();
      record.pathGroup = endpoint.pathGroup.str();
      if (tr.worstLaunch)
        record.startpoint = tr.worstLaunch->portName.str();
      record.slackNs = tr.slack.value / 1000.0;
      record.hasSlack = true;
      record.arrivalNs = tr.arrival ? tr.arrival->value / 1000.0 : 0.0;
      paths.push_back(std::move(record));
    }
  }
  return paths;
}
std::string buildFeedbackJson(StringRef top,
                              const timing::TimingResult &timing) {
  std::string out;
  llvm::raw_string_ostream os(out);
  timing::printTimingFeedbackV2(timing, top, os);
  os.flush();
  return out;
}

// Deterministic feedback signature for cycle detection: sorted
// endpoint|required|slack over violating paths plus accepted policies.
std::string
feedbackSignature(ArrayRef<StaPath> paths,
                  const std::map<std::string, std::string> &accepted) {
  std::vector<std::string> parts;
  char buf[64];
  for (const auto &path : paths) {
    if (!path.hasSlack || path.slackNs >= 0.0)
      continue;
    snprintf(buf, sizeof(buf), "%.4f|%.4f", path.arrivalNs + path.slackNs,
             path.slackNs);
    parts.push_back(path.endpoint + "|" + buf);
  }
  std::sort(parts.begin(), parts.end());
  std::string sig;
  for (const auto &part : parts) {
    sig += part;
    sig += ";";
  }
  sig += "#";
  for (const auto &kv : accepted) {
    sig += kv.first;
    sig += "=";
    sig += kv.second;
    sig += ";";
  }
  return sig;
}

bool writeFile(const std::string &path, const std::string &content) {
  std::error_code ec;
  raw_fd_ostream out(path, ec);
  if (ec) {
    errs() << "circt-synth: cannot write " << path << ": " << ec.message()
           << "\n";
    return false;
  }
  out << content;
  return true;
}

// Rank regions: violating first by worst endpoint slack (ps), then
// structural regions by worst levels, then ID. Feedback slack comes from
// the round's STA records keyed by endpoint string.
std::vector<ReportRegion> rankRegions(const std::vector<ReportRegion> &regions,
                                      ArrayRef<StaPath> feedback) {
  std::map<std::string, double> endpointSlack;
  for (const auto &path : feedback) {
    if (!path.hasSlack)
      continue;
    auto it = endpointSlack.find(path.endpoint);
    if (it == endpointSlack.end() || path.slackNs < it->second)
      endpointSlack[path.endpoint] = path.slackNs;
  }
  std::vector<ReportRegion> ranked = regions;
  std::sort(ranked.begin(), ranked.end(),
            [&](const ReportRegion &a, const ReportRegion &b) {
              auto score =
                  [&](const ReportRegion &r) -> std::pair<int, double> {
                // (0, slack) for violated-with-slack, (1, levels) for
                // structural, violated-without-slack sorts with structural by
                // levels.
                double worst = 0.0;
                bool hasSlack = false;
                for (const auto &endpoint : r.endpoints) {
                  auto it = endpointSlack.find(endpoint);
                  if (it != endpointSlack.end()) {
                    if (!hasSlack || it->second < worst)
                      worst = it->second;
                    hasSlack = true;
                  }
                }
                if (hasSlack)
                  return {0, worst};
                return {1, (double)r.worstSlackLevels.value_or(0)};
              };
              auto sa = score(a);
              auto sb = score(b);
              if (sa != sb)
                return sa < sb;
              return a.id < b.id;
            });
  return ranked;
}

const std::vector<std::string> kRefinementPolicies = {
    "reduce-delay", "booth", "compress-to-add", "fastest"};

bool runRefinementRetry(RetryRunner &runner) {
  const DatapathRetryConfig &config = runner.config;
  std::vector<CandidateResult> results;
  std::map<std::string, std::string> accepted;
  std::set<std::string> tried;
  std::set<std::string> seenSignatures;
  int usedVariants = 0;

  auto candidateFlags = [&](bool isBaseline, const std::string &report,
                            const std::string &manifest,
                            const std::string &feedback) {
    std::vector<std::string> flags = {"--datapath-refinement",
                                      "--datapath-refinement-policy=baseline",
                                      "--datapath-refinement-report=" + report};
    if (isBaseline) {
      flags.push_back("--datapath-pressure-margin-levels=-1");
      if (!config.initialFeedback.empty())
        flags.push_back("--timing-feedback=" + config.initialFeedback);
    } else {
      flags.push_back("--datapath-pressure-margin-levels=" +
                      std::to_string(config.pressureMarginLevels));
      flags.push_back("--datapath-policy-file=" + manifest);
      flags.push_back("--timing-feedback=" + feedback);
    }
    return flags;
  };

  // Round 0: refinement baseline (discover + report, no policies).
  std::string base = runner.tmpDir + "/baseline";
  CandidateResult baseline;
  if (!runner.runCandidate("baseline",
                           candidateFlags(true, base + ".report.json", "", ""),
                           baseline)) {
    errs() << "circt-synth: refinement baseline failed\n";
    return false;
  }
  results.push_back(baseline);
  size_t best = 0;

  // Fresh feedback from the baseline STA (paths with slack only).
  std::string feedbackFile = runner.tmpDir + "/feedback-r0.json";
  if (!writeFile(feedbackFile,
                 buildFeedbackJson(config.top, baseline.timing)))
    return false;
  std::vector<StaPath> feedback = toStaPaths(baseline.timing);

  for (int round = 1; round <= config.refinementIterations; ++round) {
    std::string sig = feedbackSignature(feedback, accepted);
    if (!seenSignatures.insert(sig).second) {
      errs() << "circt-synth: refinement stops: feedback signature repeats "
                "(round "
             << round << ")\n";
      break;
    }
    std::vector<ReportRegion> regions = readRefinementReport(
        runner.tmpDir + "/" + results[best].name + ".report.json");
    if (regions.empty()) {
      errs() << "circt-synth: refinement stops: no datapath regions in "
                "report (round "
             << round << ")\n";
      break;
    }
    // Propose untried single-region policies in rank order. Decided
    // regions are skipped unless currently violated (fresh information
    // re-opens them); priors are carried in every manifest.
    struct Proposal {
      std::string region;
      std::string policy;
    };
    std::map<std::string, double> endpointSlack;
    for (const auto &path : feedback) {
      if (!path.hasSlack)
        continue;
      auto it = endpointSlack.find(path.endpoint);
      if (it == endpointSlack.end() || path.slackNs < it->second)
        endpointSlack[path.endpoint] = path.slackNs;
    }
    std::vector<Proposal> proposals;
    int budget = config.maxVariants - usedVariants;
    for (const auto &region : rankRegions(regions, feedback)) {
      if ((int)proposals.size() >= std::min(4, budget))
        break;
      bool decided = accepted.count(region.id) > 0;
      bool violatedNow = false;
      for (const auto &endpoint : region.endpoints) {
        auto it = endpointSlack.find(endpoint);
        if (it != endpointSlack.end() && it->second < 0.0) {
          violatedNow = true;
          break;
        }
      }
      if (decided && !violatedNow)
        continue;
      for (const auto &policy : kRefinementPolicies) {
        if ((int)proposals.size() >= std::min(4, budget))
          break;
        auto acceptedIt = accepted.find(region.id);
        if (acceptedIt != accepted.end() && acceptedIt->second == policy)
          continue;
        std::string key = region.id + "\x01" + policy + "\x01" + sig;
        if (tried.count(key))
          continue;
        tried.insert(key);
        proposals.push_back({region.id, policy});
      }
    }
    if (proposals.empty()) {
      errs() << "circt-synth: refinement stops: no untried region policies "
                "(round "
             << round << ")\n";
      break;
    }
    // Run the round.
    std::vector<CandidateResult> roundResults;
    for (const auto &proposal : proposals) {
      std::map<std::string, std::string> manifestMap = accepted;
      manifestMap[proposal.region] = proposal.policy;
      std::string manifest = "{\"schema\": "
                             "\"circt.synth.datapath-policy.v1\", "
                             "\"regions\": [";
      bool firstEntry = true;
      for (const auto &kv : manifestMap) {
        if (!firstEntry)
          manifest += ", ";
        firstEntry = false;
        manifest +=
            "{\"id\": \"" + kv.first + "\", \"policy\": \"" + kv.second + "\"}";
      }
      manifest += "]}";
      std::string name = "ref-r" + std::to_string(round) + "-" +
                         sanitizeName(proposal.region) + "-" + proposal.policy;
      std::string manifestFile = runner.tmpDir + "/" + name + ".policy.json";
      std::string roundFeedback =
          runner.tmpDir + "/feedback-r" + std::to_string(round) + ".json";
      if (!writeFile(manifestFile, manifest) ||
          !writeFile(roundFeedback,
                      buildFeedbackJson(config.top, results[best].timing)))
        return false;
      CandidateResult result;
      if (runner.runCandidate(
              name,
              candidateFlags(false, runner.tmpDir + "/" + name + ".report.json",
                             manifestFile, roundFeedback),
              result)) {
        ++usedVariants;
        roundResults.push_back(result);
      }
    }
    if (roundResults.empty()) {
      errs() << "circt-synth: refinement stops: all round " << round
             << " candidates failed\n";
      break;
    }
    // Adopt strict improvements (baseline wins ties). Indices, not
    // pointers: results grows during adoption.
    size_t roundBest = best;
    size_t appendBase = results.size();
    for (const auto &r : roundResults) {
      results.push_back(r);
      if (isBetterRefinement(r, results[roundBest], config.slackTarget))
        roundBest = results.size() - 1;
    }
    {
      // Round table with the carried best first for context.
      std::vector<CandidateResult> display;
      display.push_back(results[best]);
      for (const auto &r : roundResults)
        display.push_back(r);
      size_t displayBest = roundBest == best ? 0 : 1 + (roundBest - appendBase);
      printRefinementTable(display, displayBest, config.slackTarget);
    }
    double before = results[best].timing.wns
                        ? results[best].timing.wns->value / 1000.0
                        : -results[best].arrivalPs / 1000.0;
    double after = results[roundBest].timing.wns
                       ? results[roundBest].timing.wns->value / 1000.0
                       : -results[roundBest].arrivalPs / 1000.0;
    bool improved = roundBest != best;
    if (improved) {
      // Carry the winning single-region decision (plus priors) forward
      // by matching the winner back to its proposal.
      const std::string &winnerName = results[roundBest].name;
      for (const auto &proposal : proposals) {
        std::string name = "ref-r" + std::to_string(round) + "-" +
                           sanitizeName(proposal.region) + "-" +
                           proposal.policy;
        if (winnerName == name)
          accepted[proposal.region] = proposal.policy;
      }
    }
    best = roundBest;
    feedback = toStaPaths(results[best].timing);
    if (results[best].timing.wns &&
        results[best].timing.wns->value >= llround(config.slackTarget * 1000.0)) {
      errs() << "circt-synth: refinement stops: slack target met (round "
             << round << ")\n";
      break;
    }
    if (!improved || (after - before) * 1000.0 < config.minImprovementPs) {
      errs() << "circt-synth: refinement stops: improvement below threshold "
                "(round "
             << round << ")\n";
      break;
    }
    if (usedVariants >= config.maxVariants) {
      errs() << "circt-synth: refinement stops: variant budget exhausted "
                "(round "
             << round << ")\n";
      break;
    }
  }

  printRefinementTable(results, best, config.slackTarget);
  return emitWinner(runner, results[best]);
}

bool circt::synth::runDatapathRetry(const DatapathRetryConfig &config) {
  if (config.top.empty()) {
    errs() << "circt-synth: --datapath-retry requires --top\n";
    return false;
  }
  if (config.liberty.empty()) {
    errs() << "circt-synth: --datapath-retry requires --liberty (STA "
              "judging needs liberty)\n";
    return false;
  }

  // Temp workspace for candidates.
  SmallString<128> tmpDir;
  if (sys::fs::createUniqueDirectory("circt-datapath-retry", tmpDir)) {
    errs() << "circt-synth: cannot create temp directory\n";
    return false;
  }
  auto cleanup = [&] {
    if (!config.keepTemps)
      sys::fs::remove_directories(tmpDir);
    else
      errs() << "circt-synth: retry temps kept in " << tmpDir << "\n";
  };

  RetryRunner runner{config, tmpDir.str().str(),
                     siblingTool(config.selfPath, "circt-opt")};
  runner.mlirContext = config.mlirContext;

  // Probe-guided refinement rounds replace the global variant list.
  if (config.refinement) {
    bool ok = runRefinementRetry(runner);
    cleanup();
    return ok;
  }

  std::vector<CandidateResult> results;
  CandidateResult baseline;
  if (!runner.runCandidate("baseline", {}, baseline)) {
    errs() << "circt-synth: datapath retry baseline failed\n";
    cleanup();
    return false;
  }
  results.push_back(baseline);
  for (int i = 0; i < config.maxVariants && i < (int)kVariants.size(); ++i)
    if (CandidateResult result;
        runner.runCandidate(kVariants[i].first, kVariants[i].second, result))
      results.push_back(result);

  // Select: min area among slack-satisfying (else max slack; clockless
  // judges by min arrival). Baseline wins all ties: never regresses.
  // Timing compares on integral picoseconds.
  int64_t targetPs = llround(config.slackTarget * 1000.0);
  auto isSatisfying = [&](const CandidateResult &r) {
    return r.timing.wns && r.timing.wns->value >= targetPs;
  };
  bool anySatisfying = false;
  for (const auto &r : results)
    anySatisfying |= isSatisfying(r);
  const CandidateResult *best = &results.front();
  for (size_t i = 1; i < results.size(); ++i) {
    const CandidateResult &r = results[i];
    if (anySatisfying) {
      bool bestSat = isSatisfying(*best);
      if (isSatisfying(r) && (!bestSat || r.instances < best->instances))
        best = &r;
    } else if (r.timing.wns && best->timing.wns) {
      if (r.timing.wns->value > best->timing.wns->value)
        best = &r;
    } else if (!r.timing.wns && !best->timing.wns) {
      if (r.arrivalPs < best->arrivalPs)
        best = &r;
    }
  }

  errs() << "circt-synth: datapath retry results (slack target "
         << config.slackTarget << "ns):\n";
  for (const auto &r : results) {
    errs() << "  [" << (&r == best ? "BEST" : "    ") << "] " << r.name;
    if (!r.flags.empty())
      errs() << " (" << r.flags << ")";
    errs() << ": inst=" << r.instances;
    if (r.timing.wns)
      errs() << " slack=" << (r.timing.wns->value / 1000.0) << "ns";
    errs() << " arrival=" << (r.arrivalPs / 1000.0) << "ns\n";
  }

  bool ok = emitWinner(runner, *best);
  cleanup();
  return ok;
}
