#!/usr/bin/env python3
"""Benchmark suite driver.

    suite.py track1|track2 --box my-box.json \
        --systems superfluid-llamacpp,llama-server [--scenarios worker_kill,daemon_restart] [--out DIR]

Runs each (system, scenario) cell in turn: start the system fresh, run the
scenario, stop it, cool down, write one JSON per cell plus a manifest. Tables
come from those files only (compare.py), never from console output.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import systems  # noqa: E402
import track1  # noqa: E402
import track2  # noqa: E402

TRACKS = {"track1": track1, "track2": track2}


def lock(path):
    """One driver per box: a second one would share GPUs and ports."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = open(path).read().strip()
        if pid and subprocess.run(["kill", "-0", pid], capture_output=True).returncode == 0:
            sys.exit(f"another suite driver (pid {pid}) holds {path}")
        os.unlink(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def kill_strays():
    """Servers left by an earlier, killed driver would answer our probes."""
    for port in systems.PORTS.values():
        if common.port_open(port):
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
            for pid in out:
                subprocess.run(["kill", "-9", pid])
            print(f"killed stray listener(s) {out} on {port}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=sorted(TRACKS))
    ap.add_argument("--box", required=True)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cooldown", type=int, default=60, help="seconds idle between cells")
    ap.add_argument("--quick", action="store_true", help="smaller scenario sizes, for a dry run of the harness")
    args = ap.parse_args()

    box = systems.load_box(args.box)
    mod = TRACKS[args.track]
    names = [s for s in args.systems.split(",") if s]
    scen = [s for s in args.scenarios.split(",") if s] or list(mod.SCENARIOS)
    out = args.out or os.path.join(box["root"], "results", f"{time.strftime('%Y-%m-%d')}-{box['box']}-{args.track}")
    os.makedirs(out, exist_ok=True)
    lock(os.path.join(box["root"], ".suite.lock"))
    try:
        kill_strays()
        stamp = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(box["superfluid"]))), ".suite-rev")
        rev = open(stamp).read().strip() if os.path.exists(stamp) else None
        manifest = {"track": args.track, "box": box["box"], "chip": box.get("chip"), "host": socket.gethostname(),
                    "versions": dict(systems.versions(box), superfluid=rev), "systems": names, "scenarios": scen,
                    "quick": args.quick, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
        for sc in scen:
            for name in names:
                path = os.path.join(out, f"{sc}--{name}.json")
                if os.path.exists(path):
                    print(f"[{sc} / {name}] already in {path}, skipping", flush=True)
                    continue
                print(f"[{sc} / {name}] start  therm: {common.thermal()}", flush=True)
                t0 = time.time()
                try:
                    res = mod.SCENARIOS[sc](name, box, out, quick=args.quick)
                except Exception as e:  # noqa: BLE001
                    res = {"verdict": "error", "error": repr(e), "trace": traceback.format_exc()[-3000:]}
                res.update({"scenario": sc, "system": name, "seconds": round(time.time() - t0, 1),
                            "therm_after": common.thermal(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
                json.dump(res, open(path, "w"), indent=1)
                print(f"[{sc} / {name}] {res['verdict']}: {res.get('headline', res.get('reason', res.get('error', '')))}", flush=True)
                time.sleep(args.cooldown)
        manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    finally:
        os.unlink(os.path.join(box["root"], ".suite.lock"))
    print("SUITE-DONE", out, flush=True)


if __name__ == "__main__":
    main()
