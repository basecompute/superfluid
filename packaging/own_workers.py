#!/usr/bin/env python3
"""Check that an installed package runs the runtime workers it carries.

    superfluid runtimes --json | python packaging/own_workers.py <release-dir>

<release-dir> is the release inside the package. packaging/try-install.sh
runs this on what the installed `superfluid` says of its runtimes; it is a
file of its own so that tests can give it both kinds of release.

A release has a worker for a runtime in one of two ways, and `runtimes` names
which as the worker's source:

  shipped with superfluid   an adapter of its own, in libexec/superfluid
                            (the macOS and Linux releases' llama.cpp and MLX)
  built in                  linked into the daemon, and run by the package's
                            bin/superfluid-workerd (the Windows release's
                            llama.cpp)

Either way the worker must be a file of this package, and llama.cpp, which
every release serves with, must be among them. A worker from anywhere else
(an installed runtime, one named in the environment) is not the package's
and is not looked at.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Source::name() in crates/superfluid-daemon/src/runtimes.rs, and where under
# the release a worker of that source is.
OWN = {"shipped with superfluid": Path("libexec") / "superfluid", "built in": Path("bin")}


class NotOwn(Exception):
    pass


def real(path) -> str:
    """A path as one to compare: links and short names resolved, and on
    Windows without regard to case or to which way the slashes lean."""
    return os.path.normcase(os.path.realpath(path))


def own_workers(rows: list, where) -> dict[str, str]:
    """The package's own workers in `superfluid runtimes --json`, by runtime id
    with each one's source; NotOwn when one is not in the release at `where`,
    or llama.cpp has none."""
    found = {}
    for row in rows:
        worker = row.get("worker") or {}
        folder = OWN.get(worker.get("source"))
        if folder is None:
            continue
        rid, path = row["id"], worker.get("path") or ""
        if os.path.dirname(real(path)) != real(Path(where) / folder):
            raise NotOwn(f"the {rid} worker ({worker['source']}) is {path}, not in {Path(where) / folder}")
        if not os.path.isfile(path):
            raise NotOwn(f"the {rid} worker ({worker['source']}) {path} is not there")
        found[rid] = worker["source"]
    if "llamacpp" not in found:
        said = {row.get("id"): (row.get("worker") or {}).get("source") for row in rows}
        raise NotOwn(f"no llama.cpp worker of the package's own: {said}")
    # The adapters and what goes with them (on Windows the launcher and the
    # Visual C++ runtime) are carried whichever way the workers are found.
    adapters = Path(where) / OWN["shipped with superfluid"]
    if not adapters.is_dir() or not any(adapters.iterdir()):
        raise NotOwn(f"{adapters} holds nothing: the package lost what the release ships there")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: superfluid runtimes --json | {argv[0]} <release-dir>", file=sys.stderr)
        return 2
    try:
        found = own_workers(json.load(sys.stdin.buffer), argv[1])
    except NotOwn as e:
        print(f"own_workers.py: {e}", file=sys.stderr)
        return 1
    print("workers:", ", ".join(f"{rid} ({source})" for rid, source in sorted(found.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
