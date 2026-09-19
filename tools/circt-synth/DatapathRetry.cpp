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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace circt::synth;
using namespace llvm;

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

struct CandidateResult {
  std::string name;
  std::string flags;
  bool ok = false;
  int64_t instances = -1;
  // Clocked judging (when the SDC has clocks).
  bool hasSlack = false;
  double slackNs = 0.0;
  double arrivalNs = 0.0;
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

} // namespace

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

  std::string circtOpt = siblingTool(config.selfPath, "circt-opt");

  // Candidate synth commands share one builder.
  auto runCandidate = [&](StringRef name, ArrayRef<std::string> variantFlags,
                          CandidateResult &result) -> bool {
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
      tcl += hasClocks ? "report_checks -path_delay max\n"
                       : "report_checks -path_delay max -unconstrained\n";
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
    std::optional<double> slack;
    std::optional<double> arrival;
    SmallVector<StringRef> lines;
    StringRef(log).split(lines, '\n');
    for (StringRef line : lines) {
      if (auto s = parseTrailingNumber(line, "slack"))
        slack = !slack ? s : std::min(*slack, *s);
      if (auto a = parseTrailingNumber(line, "data arrival time"))
        arrival = !arrival ? a : std::max(*arrival, *a);
    }
    if (!arrival) {
      errs() << "circt-synth: retry candidate '" << name
             << "' STA unparsable (skipped)\n";
      return false;
    }
    result.arrivalNs = *arrival;
    if (slack) {
      result.hasSlack = true;
      result.slackNs = *slack;
    }

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
  };

  std::vector<CandidateResult> results;
  CandidateResult baseline;
  if (!runCandidate("baseline", {}, baseline)) {
    errs() << "circt-synth: datapath retry baseline failed\n";
    cleanup();
    return false;
  }
  results.push_back(baseline);
  for (int i = 0; i < config.maxVariants && i < (int)kVariants.size(); ++i)
    if (CandidateResult result;
        runCandidate(kVariants[i].first, kVariants[i].second, result))
      results.push_back(result);

  // Select: min area among slack-satisfying (else max slack; clockless
  // judges by min arrival). Baseline wins all ties: never regresses.
  auto isSatisfying = [&](const CandidateResult &r) {
    return r.hasSlack && r.slackNs >= config.slackTarget;
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
    } else if (r.hasSlack && best->hasSlack) {
      if (r.slackNs > best->slackNs)
        best = &r;
    } else if (!r.hasSlack && !best->hasSlack) {
      if (r.arrivalNs < best->arrivalNs)
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
    if (r.hasSlack)
      errs() << " slack=" << r.slackNs << "ns";
    errs() << " arrival=" << r.arrivalNs << "ns\n";
  }

  // Emit the winner.
  std::string winnerMlir = (tmpDir + ("/" + best->name + ".mlir")).str();
  if (config.outputFile == "-") {
    auto out = MemoryBuffer::getFile(winnerMlir);
    if (!out) {
      errs() << "circt-synth: cannot read winner\n";
      cleanup();
      return false;
    }
    outs() << out.get()->getBuffer();
  } else if (sys::fs::copy_file(winnerMlir, config.outputFile)) {
    errs() << "circt-synth: cannot write " << config.outputFile << "\n";
    cleanup();
    return false;
  }
  if (!config.userSdcOutput.empty()) {
    std::string winnerSdc = (tmpDir + ("/" + best->name + ".sdc")).str();
    if (sys::fs::copy_file(winnerSdc, config.userSdcOutput))
      errs() << "circt-synth: warning: cannot write " << config.userSdcOutput
             << "\n";
  }
  cleanup();
  return true;
}
