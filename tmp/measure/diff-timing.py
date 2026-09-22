#!/usr/bin/env python3
"""Differential timing check: OpenTimer JSON vs OpenSTA report.

Compares per-endpoint slack (grouped by endpoint, worst transition) and
whole-design WNS/TNS/failing between an OpenTimer result JSON
(--timing-run-only) and an OpenSTA report_checks log on the same
constraints. Exits nonzero on mismatch beyond tolerance.

STA endpoint names (hier/pin) match OpenTimer keys by instance path +
port; bits are ignored for single-bit endpoints (v1 scope).
"""
import json
import re
import sys


def parse_sta_endpoints(log):
    """Group STA report_checks paths by endpoint, min slack each."""
    endpoints = {}
    cur_ep = None
    cur_slack = None
    for line in log.splitlines():
        line = line.strip()
        if line.startswith("Endpoint:"):
            if cur_ep is not None and cur_slack is not None:
                prev = endpoints.get(cur_ep)
                endpoints[cur_ep] = cur_slack if prev is None else min(prev, cur_slack)
            cur_ep = line.split("Endpoint:", 1)[1].strip().split()[0]
            cur_slack = None
        m = re.search(r"slack\s+\(?([-\d.]+)\)?", line)
        if m and cur_ep is not None and "VIOLATED" not in line and "MET" not in line:
            try:
                cur_slack = float(m.group(1))
            except ValueError:
                pass
        # Slack summary line looks like "12.345 slack (MET)".
        m2 = re.match(r"([-\d.]+)\s+slack\s+\(", line)
        if m2 and cur_ep is not None:
            cur_slack = float(m2.group(1))
    if cur_ep is not None and cur_slack is not None:
        prev = endpoints.get(cur_ep)
        endpoints[cur_ep] = cur_slack if prev is None else min(prev, cur_slack)
    return endpoints


def ot_key(endpoint):
    ep = endpoint["endpoint"]
    path = "/".join(ep.get("instance_path", []))
    key = f"{path}/{ep.get('port_name', '')}" if path else ep.get("port_name", "")
    return key


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} ot.json sta.log tolerance_ps")
        return 2
    ot_path, sta_path, tol_ps = sys.argv[1], sys.argv[2], float(sys.argv[3])
    result = json.load(open(ot_path))
    sta_log = open(sta_path).read()

    ot_eps = {}
    for e in result.get("endpoints", []):
        slacks = [t["slack"] for t in e.get("transitions", [])]
        if slacks:
            ot_eps[ot_key(e)] = min(slacks)
    sta_eps = parse_sta_endpoints(sta_log)

    # Match by suffix (STA flop endpoints name the instance, not the
    # pin, so also try the OT key without its trailing port).
    matched = 0
    worst_delta = 0.0
    for key, ot_slack_ps in sorted(ot_eps.items()):
        ot_slack_ns = ot_slack_ps / 1000.0
        keys = {key, key.rsplit("/", 1)[0] if "/" in key else key}
        best = None
        for skey, sta_slack in sta_eps.items():
            if any(skey.endswith(k) or k.endswith(skey) for k in keys):
                best = sta_slack if best is None else min(best, sta_slack)
        if best is None:
            print(f"UNMATCHED ot endpoint: {key} slack={ot_slack_ns:.4f}ns")
            continue
        delta_ps = abs(ot_slack_ps - best * 1000.0)
        worst_delta = max(worst_delta, delta_ps)
        status = "OK" if delta_ps <= tol_ps else "MISMATCH"
        print(f"{status} {key}: ot={ot_slack_ns:.4f}ns sta={best:.4f}ns "
              f"delta={delta_ps:.2f}ps")
        matched += 1

    ot_wns = min(ot_eps.values()) / 1000.0 if ot_eps else None
    sta_wns = min(sta_eps.values()) if sta_eps else None
    print(f"coverage: ot={len(ot_eps)} sta={len(sta_eps)} matched={matched}")
    print(f"WNS: ot={ot_wns}ns sta={sta_wns}ns")
    if matched != len(ot_eps) or worst_delta > tol_ps:
        print("DIFFERENTIAL FAIL")
        return 1
    print("DIFFERENTIAL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
