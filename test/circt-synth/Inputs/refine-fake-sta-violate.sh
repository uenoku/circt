#!/bin/sh
# Fake `sta` for the no-improvement refinement test: every candidate
# violates like the baseline, so the baseline must win ties and be
# emitted exactly.
slack="-5.0000   slack (VIOLATED)"
arrival="15.0000   data arrival time"
printf 'Startpoint: u_sub/r0/Q\nEndpoint: u_sub/r0/D\nPath Group: clk\nPath Type: max\n%s\n%s\n' \
  "$slack" "$arrival"
