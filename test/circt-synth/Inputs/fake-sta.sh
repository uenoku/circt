#!/bin/sh
# Fake `sta` for datapath-retry lit tests: canned report_checks output with
# variant-dependent slack (the .tcl path embeds the variant name), so the
# loop's judge logic is tested hermetically without OpenSTA.
case "$*" in
  *force-booth*)
    slack="2.0000   slack (MET)"
    arrival="8.0000   data arrival time"
    ;;
  *compress-to-add*)
    slack="-1.0000   slack (VIOLATED)"
    arrival="11.0000   data arrival time"
    ;;
  *)
    slack="-5.0000   slack (VIOLATED)"
    arrival="15.0000   data arrival time"
    ;;
esac
printf 'Startpoint: a\nEndpoint: z\nPath Group: clk\nPath Type: max\n%s\n%s\n' \
  "$slack" "$arrival"
