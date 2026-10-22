#!/usr/bin/env python3
"""The nightly runtimes job's two decisions: which upstream release to test,
and how to record one that passed.

    nightly_runtimes.py llamacpp pick [--requested bNNNN]
    nightly_runtimes.py llamacpp append bNNNN
    nightly_runtimes.py mlx pick [--requested X.Y.Z]
    nightly_runtimes.py mlx append X.Y.Z --report <pip report.json>

`pick` prints `version=<v>` for the newest upstream release the adapter's
recipe does not list yet, or `version=` when there is nothing new (or when
`--requested` names a release already listed); the workflow reads it from
$GITHUB_OUTPUT. For MLX a release is the pair of mlx and mlx-lm, and `pick`
also prints `mlx_lm=<v>`: a new mlx-lm on the newest mlx is tested as well
(a requested mlx gets whichever mlx-lm pip resolves, `mlx_lm=`). A newest
release whose PR was closed unmerged is not picked again: someone said no to
it, and only a `--requested` one asks again.
`append` adds a release to the adapter's recipe as a tested one, newest
first, so a human reviews a PR rather than typing digests. For llama.cpp
that is the same asset names the previous tested build carries, with the
size and SHA-256 GitHub lists for each. For MLX it is the whole environment
pip resolved for the version on the runner, read from pip's install report
(`pip install --dry-run --report`): every distribution's wheel, with the
size and SHA-256 PyPI lists for it, which must agree with the digest pip
reported; an mlx already listed has its entry replaced, for the newer mlx-lm.

Only the standard library; GITHUB_TOKEN is used when set, for the API rate
limit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECIPES = {
    "llamacpp": ROOT / "crates/superfluid-adapters/superfluid-adapter-llamacpp/recipe.json",
    "mlx": ROOT / "crates/superfluid-adapters/superfluid-adapter-mlx/recipe.json",
}
LLAMACPP_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
LLAMACPP_RELEASE = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{version}"
MLX_PYPI = "https://pypi.org/pypi/mlx/json"
MLX_LM_PYPI = "https://pypi.org/pypi/mlx-lm/json"
# The record jobs' PRs come from one branch per runtime, newest first in this listing.
CLOSED_PRS = "https://api.github.com/repos/{repo}/pulls?state=closed&head={owner}:nightly/{runtime}&per_page=100"
PYPI_RELEASE = "https://pypi.org/pypi/{name}/{version}/json"
BUILD = re.compile(r"^b(\d+)$")
# A PyPI release as mlx and mlx-lm number theirs (PEP 440's public versions, short of epochs).
PYPI_VERSION = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "superfluid-nightly"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url}: HTTP {e.code}") from None


def recipe(runtime: str) -> dict:
    return json.loads(RECIPES[runtime].read_text(encoding="utf-8"))


def tested_versions(runtime: str) -> list[str]:
    return [t["version"] for t in recipe(runtime).get("tested", [])]


def build_number(tag: str) -> int:
    m = BUILD.fullmatch(tag)
    return int(m.group(1)) if m else -1


def newest_llamacpp_build() -> dict:
    """The newest `b<N>` release that carries the macOS arm64 archive, which
    every tested build has; llama.cpp also tags semver releases, which are
    not builds of the library."""
    releases = fetch_json(LLAMACPP_RELEASES)
    builds = [
        r
        for r in releases
        if build_number(r["tag_name"]) >= 0
        and not r.get("draft")
        and any(a["name"] == f"llama-{r['tag_name']}-bin-macos-arm64.tar.gz" for a in r.get("assets", []))
    ]
    if not builds:
        raise SystemExit("no llama.cpp build with a macOS arm64 archive in the newest 30 releases")
    return max(builds, key=lambda r: build_number(r["tag_name"]))


def pr_title(runtime: str, version: str, mlx_lm: str = "") -> str:
    """The title the record job gives the PR that adds `version`; the two must agree."""
    return f"{runtime}: tested with {version}" + (f" and mlx-lm {mlx_lm}" if mlx_lm else "")


def declined(runtime: str, version: str, mlx_lm: str = "") -> bool:
    """Whether a PR adding `version` was closed without being merged. Outside
    Actions (no GITHUB_REPOSITORY) there is no repository to ask, and nothing was."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return False
    pulls = fetch_json(CLOSED_PRS.format(repo=repo, owner=repo.split("/")[0], runtime=runtime))
    title = pr_title(runtime, version, mlx_lm)
    return any(p["title"] == title and not p.get("merged_at") for p in pulls)


def checked(runtime: str, version: str, source: str) -> str:
    """`version`, if it looks like one of `runtime`'s; it goes on into job names,
    PR titles and pip requirements, so anything else stops the run here."""
    pattern = BUILD if runtime == "llamacpp" else PYPI_VERSION
    if not pattern.fullmatch(version):
        shape = "bNNNN" if runtime == "llamacpp" else "a PEP 440 version, X.Y.Z"
        raise SystemExit(f"{source} {version!r} is not a release of {runtime} ({shape})")
    return version


def pick(runtime: str, requested: str | None) -> dict[str, str]:
    """The workflow's outputs: `version`, and for MLX `mlx_lm`, all empty when there is nothing to test."""
    out = {"version": ""} if runtime == "llamacpp" else {"version": "", "mlx_lm": ""}
    if requested:
        checked(runtime, requested, "the requested release")
        if requested not in tested_versions(runtime):
            out["version"] = requested
        return out
    if runtime == "llamacpp":
        newest = newest_llamacpp_build()["tag_name"]
        have = max((build_number(v) for v in tested_versions(runtime)), default=-1)
        if build_number(newest) <= have:
            return out
        mlx_lm = ""
    else:
        # A release of either package is a change the adapter's tests should see.
        newest = checked(runtime, fetch_json(MLX_PYPI)["info"]["version"], "PyPI's newest mlx")
        mlx_lm = checked(runtime, fetch_json(MLX_LM_PYPI)["info"]["version"], "PyPI's newest mlx-lm")
        if any(t["version"] == newest and t.get("mlx_lm") == mlx_lm for t in recipe(runtime).get("tested", [])):
            return out
        out["mlx_lm"] = mlx_lm
    if declined(runtime, newest, mlx_lm):
        print(f"note: the PR titled {pr_title(runtime, newest, mlx_lm)!r} was closed unmerged; not proposing it again", file=sys.stderr)
        return {k: "" for k in out}
    out["version"] = newest
    return out


def append_llamacpp(version: str) -> None:
    path = RECIPES["llamacpp"]
    r = recipe("llamacpp")
    tested = r.setdefault("tested", [])
    if any(t["version"] == version for t in tested):
        print(f"{version} is already a tested build", file=sys.stderr)
        return
    if not tested:
        raise SystemExit("the recipe lists no tested build to take the asset names from")
    previous = tested[0]
    release = fetch_json(LLAMACPP_RELEASE.format(version=version))
    by_name = {a["name"]: a for a in release.get("assets", [])}
    assets, missing = [], []
    for old in previous["assets"]:
        name = old["name"].replace(previous["version"], version)
        a = by_name.get(name)
        digest = (a or {}).get("digest") or ""
        if not a or not digest.startswith("sha256:"):
            missing.append(name)
            continue
        assets.append({"name": name, "size": a["size"], "sha256": digest[len("sha256:") :]})
    if missing:
        # The archives the tests ran on must be there; a CUDA runtime archive
        # dropped upstream is reported and left out.
        required = [n for n in missing if n.endswith(("bin-macos-arm64.tar.gz", "bin-ubuntu-x64.tar.gz"))]
        if required:
            raise SystemExit(f"{version} lacks, or lists no SHA-256 for, {', '.join(required)}")
        print(f"note: {version} lacks {', '.join(missing)}; left out", file=sys.stderr)
    entry = {"version": version, "published": release["published_at"][:10], "assets": assets}
    tested.insert(0, entry)
    path.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded {version} ({release['published_at'][:10]}) with {len(assets)} assets in {path.relative_to(ROOT)}")


def canonical(name: str) -> str:
    """A project name as PyPI's URLs spell it; the recipe keeps the spelling
    the distribution gives itself (`PyYAML`, `Jinja2`) and orders by this."""
    return re.sub(r"[-_.]+", "-", name).lower()


MACOS_TAG = re.compile(r"-macosx_(\d+)_(\d+)_arm64\.whl$")
# The oldest macOS the adapter runs on: a wheel built for it installs there and
# on every later one, where a wheel built for the runner's own macOS would not.
MACOS_FLOOR = (14, 0)


def floored(file: str, files: list[str]) -> str:
    """The wheel to pin for `file`: the same wheel built for the oldest macOS
    at or above the floor that PyPI offers, since pip on a newer runner or
    machine resolves the newest, which an older Mac cannot install."""
    m = MACOS_TAG.search(file)
    if not m:
        return file
    stem = file[: m.start()]
    offered = []
    for f in files:
        fm = MACOS_TAG.search(f)
        if fm and f[: fm.start()] == stem:
            offered.append(((int(fm.group(1)), int(fm.group(2))), f))
    at_or_above = sorted(v for v in offered if v[0] >= MACOS_FLOOR)
    return at_or_above[0][1] if at_or_above else file


def pypi_wheel(name: str, version: str, file: str) -> tuple[str, dict]:
    """The wheel pinned for `file` (see `floored`) and the size and SHA-256 PyPI lists for it."""
    listing = fetch_json(PYPI_RELEASE.format(name=name, version=version))
    urls = listing.get("urls", [])
    file = floored(file, [u["filename"] for u in urls])
    for u in urls:
        if u["filename"] == file:
            return file, {"size": u["size"], "sha256": u["digests"]["sha256"]}
    raise SystemExit(f"PyPI lists no {file} for {name} {version}")


def append_mlx(version: str, report_path: Path) -> None:
    path = RECIPES["mlx"]
    r = recipe("mlx")
    tested = r.setdefault("tested", [])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    wheels, mlx_lm = [], None
    for item in report.get("install", []):
        meta = item["metadata"]
        name, ver = meta["name"], meta["version"]
        url = item["download_info"]["url"]
        file = url.rsplit("/", 1)[-1]
        if not file.endswith(".whl"):
            raise SystemExit(f"{name} {ver} resolved to {file}, not a wheel; the recipe pins wheels only")
        if canonical(name) == "mlx" and ver != version:
            raise SystemExit(f"the report resolved mlx {ver}, not {version}")
        if canonical(name) == "mlx-lm":
            mlx_lm = ver
        pinned_file, pinned = pypi_wheel(canonical(name), ver, file)
        reported = item["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
        if pinned_file == file and reported and reported != pinned["sha256"]:
            raise SystemExit(f"{file}: the SHA-256 pip reported differs from the one PyPI lists")
        if pinned_file != file:
            print(f"note: {file} pinned as {pinned_file}, the same wheel built for macOS {MACOS_FLOOR[0]}", file=sys.stderr)
        wheels.append({"name": name, "version": ver, "file": pinned_file, "size": pinned["size"], "sha256": pinned["sha256"]})
    if not any(canonical(w["name"]) == "mlx" for w in wheels) or mlx_lm is None:
        raise SystemExit("the report holds no mlx or no mlx-lm wheel")
    wheels.sort(key=lambda w: canonical(w["name"]))
    entry = {"version": version, "mlx_lm": mlx_lm, "wheels": wheels}
    # One entry per mlx, which the adapter looks up by version: a newer mlx-lm
    # tested on an mlx already listed takes its entry's place.
    at = next((i for i, t in enumerate(tested) if t["version"] == version), None)
    if at is not None and tested[at].get("mlx_lm") == mlx_lm:
        print(f"mlx {version} with mlx-lm {mlx_lm} is already a tested version", file=sys.stderr)
        return
    if at is None:
        tested.insert(0, entry)
    else:
        tested[at] = entry
    path.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded mlx {version} (mlx-lm {mlx_lm}) with {len(wheels)} wheels in {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runtime", choices=["llamacpp", "mlx"])
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pick", help="the newest upstream release the recipe does not list, as version=<v>")
    p.add_argument("--requested", default="", help="test this release instead of the newest")
    a = sub.add_parser("append", help="record a release that passed as a tested one")
    a.add_argument("version")
    a.add_argument("--report", help="mlx: pip's install report (--dry-run --report) for the version on the runner")
    args = ap.parse_args()
    if args.action == "pick":
        for key, value in pick(args.runtime, args.requested.strip() or None).items():
            print(f"{key}={value}")
        return 0
    if args.runtime == "llamacpp":
        append_llamacpp(args.version)
        return 0
    if not args.report:
        raise SystemExit("mlx append needs --report, pip's install report for the version")
    append_mlx(args.version, Path(args.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
