#!/usr/bin/env python3
"""Tables from a results directory, and nothing else.

    compare.py <results-dir> [<results-dir> ...] > report.md

One pass/fail grid (scenario x system) and one section per scenario with its
condition and each system's headline numbers. Every cell links back to the
JSON file it came from."""
import glob
import json
import os
import sys

MARK = {"pass": "pass", "fail": "**FAIL**", "unsupported": "n/a", "error": "error"}


def main():
    cells, manifests = {}, []
    for d in sys.argv[1:]:
        m = os.path.join(d, "manifest.json")
        if os.path.exists(m):
            manifests.append((d, json.load(open(m))))
        for f in sorted(glob.glob(os.path.join(d, "*--*.json"))):
            r = json.load(open(f))
            r["_file"] = os.path.relpath(f)
            cells[(r["scenario"], r["system"])] = r
    scen = list(dict.fromkeys(s for s, _ in cells))
    syst = list(dict.fromkeys(n for _, n in cells))
    for d, m in manifests:
        v = m.get("versions", {})
        print(f"- `{d}`: {m.get('box')} ({m.get('chip')}), {m.get('started')} -> {m.get('finished', 'unfinished')}; "
              + ", ".join(f"{k} {val}" for k, val in v.items() if k != "captured") + (" (QUICK sizes)" if m.get("quick") else ""))
    print()
    print("| scenario | " + " | ".join(syst) + " |")
    print("|---|" + "---|" * len(syst))
    for s in scen:
        row = [MARK.get(cells[(s, n)]["verdict"], cells[(s, n)]["verdict"]) if (s, n) in cells else "" for n in syst]
        print(f"| {s} | " + " | ".join(row) + " |")
    for s in scen:
        print(f"\n### {s}\n")
        cond = next((cells[(s, n)].get("condition") for n in syst if (s, n) in cells and cells[(s, n)].get("condition")), None)
        if cond:
            print(f"Pass condition: {cond}.\n")
        for n in syst:
            r = cells.get((s, n))
            if not r:
                continue
            text = r.get("headline") or r.get("reason") or r.get("error", "")
            print(f"- **{n}** ({MARK.get(r['verdict'], r['verdict'])}): {text} — `{r['_file']}`")


if __name__ == "__main__":
    main()
