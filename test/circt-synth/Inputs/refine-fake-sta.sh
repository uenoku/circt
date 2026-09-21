#!/bin/sh
# Fake `sta` for probe-guided refinement lit tests: canned report_checks
# output keyed by candidate name (embedded in the .tcl path argument).
# reduce-delay meets, booth improves but misses, everything else violates
# like the baseline, so the judge's whole-design objective is exercised
# hermetically without OpenSTA.
case "$*" in
  *reduce-delay*)
    slack="2.0000   slack (MET)"
    arrival="8.0000   data arrival time"
    ;;
  *booth*)
    slack="-1.0000   slack (VIOLATED)"
    arrival="11.0000   data arrival time"
    ;;
  *)
    slack="-5.0000   slack (VIOLATED)"
    arrival="15.0000   data arrival time"
    ;;
esac
printf 'Startpoint: u_sub/r0/Q\nEndpoint: u_sub/r0/D\nPath Group: clk\nPath Type: max\n%s\n%s\n' \
  "$slack" "$arrival"
