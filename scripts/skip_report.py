#!/usr/bin/env python3
"""Count what a `cargo test` run actually ran, apart from what it skipped.

A fixture-gated test that finds no fixture prints `SKIP: <why>` and returns,
so it passes: a suite of real-model cases is green in a fraction of a second
on a runner with no model. cargo hides a passing test's output, so the skip
is not even in the log. This reads the run's output and says, per run, how
many tests passed, how many SKIP notices those passes printed, and why.

    cargo test -p <crate> -- --nocapture 2>&1 | tools/skip_report.py "<label>"

- Every line is passed through, so the log is the run's own.
- The summary goes to stdout and, in CI, to the job's step summary; each
  distinct skip reason becomes one `::warning::` annotation.
- `SUPERFLUID_REQUIRE_FIXTURES=1` makes any skip a failure: for a job whose
  runner is provisioned to run the real-model cases, so a runner that lost a
  fixture fails instead of passing.
- Exits non-zero when the run itself failed (a failed test, a build error,
  no test result at all), read from cargo's output. That reading is a second
  line of defence, not the first: run the pipe under `set -o pipefail`, so
  that cargo's own exit status fails the step whatever it printed.
"""

import collections
import os
import re
import sys

SKIP = re.compile(r"(?<![A-Za-z_])SKIP(?=[: ])[: ]*(.*)")
RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")
BROKEN = re.compile(r"^error: (test failed|could not compile|\d+ targets? failed)|test result: FAILED")
TRAILING = re.compile(r"\s*test \S+ \.\.\. \w+$")

def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "tests"
    passed = failed = ignored = suites = 0
    broken = False
    skips: "collections.Counter[str]" = collections.Counter()
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        text = line.rstrip("\n")
        m = RESULT.search(text)
        if m:
            suites += 1
            passed += int(m.group(1))
            failed += int(m.group(2))
            ignored += int(m.group(3))
        if BROKEN.search(text):
            broken = True
        m = SKIP.search(text)
        if m:
            why = TRAILING.sub("", m.group(1)).strip(" —-:") or "no reason given"
            skips[why] += 1
    sys.stdout.flush()

    skipped = sum(skips.values())
    summary = f"{passed} passed, {skipped} SKIP notice(s) among them; {failed} failed; {ignored} ignored"
    print(f"[skips] {label}: {summary}")
    for why, n in skips.most_common():
        print(f"[skips]   {n} x SKIP: {why}")
        print(f"::warning::{label}: SKIPPED ({n}) — {why}")

    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"\n### {label}\n\n")
            f.write(f"{passed} passed, **{skipped} SKIP notice(s) among them**; {failed} failed; {ignored} ignored.\n\n")
            if skips:
                f.write("| notices | why |\n|---:|---|\n")
                for why, n in skips.most_common():
                    f.write(f"| {n} | {why} |\n")
                f.write("\n")

    if suites == 0:
        print(f"::error::{label}: no test result in the output — nothing ran")
        return 1
    if failed or broken:
        return 1
    if skipped and os.environ.get("SUPERFLUID_REQUIRE_FIXTURES", "") not in ("", "0"):
        print(f"::error::{label}: {skipped} SKIP notice(s) and SUPERFLUID_REQUIRE_FIXTURES is set — a fixture this job needs is missing")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
