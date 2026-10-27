"""Build the npm packages of one superfluid release.

    python3 packaging/npm/build_packages.py <assets-dir> <out-dir> --version <v>

<assets-dir> holds the release's three tarballs and their .sha256 files. Writes
four tarballs to <out-dir>, each ready for `npm publish <tarball>`:

    basecompute-superfluid-darwin-arm64-<v>.tgz   the release for macOS on Apple silicon
    basecompute-superfluid-linux-x64-<v>.tgz      ... for Linux on x86-64 (glibc)
    basecompute-superfluid-linux-arm64-<v>.tgz    ... for Linux on arm64 (glibc)
    basecompute-superfluid-<v>.tgz                the `superfluid` commands

Each platform package carries the release tree in release/ (packaging/release.py
stages it, adding the installed-by file), restricted by os, cpu and libc so npm
only installs it where it runs. The main package lists all three as optional
dependencies, so each machine downloads only its own (as esbuild ships), and
its commands run the binaries from that one (packaging/npm/launch.js).
`superfluid` is taken on npm, hence the @basecompute scope. Needs npm.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import release  # noqa: E402

SCOPE = "@basecompute"
MAIN = f"{SCOPE}/superfluid"
INSTALLED_BY = f"npm: npm install --global {MAIN}@latest"
REPOSITORY = {"type": "git", "url": "git+https://github.com/basecompute/superfluid.git"}
HOMEPAGE = "https://superfluid.sh"
DESCRIPTION = "One serving daemon for local LLM inference, across runtimes and machines"
# Release target -> the npm package for it. Keep in step with PACKAGES in launch.js.
TARGETS = {
    "aarch64-apple-darwin": {"name": f"{SCOPE}/superfluid-darwin-arm64", "os": ["darwin"], "cpu": ["arm64"], "what": "macOS on Apple silicon"},
    "x86_64-unknown-linux-gnu": {"name": f"{SCOPE}/superfluid-linux-x64", "os": ["linux"], "cpu": ["x64"], "libc": ["glibc"], "what": "Linux on x86-64 (glibc 2.28 or later)"},
    "aarch64-unknown-linux-gnu": {"name": f"{SCOPE}/superfluid-linux-arm64", "os": ["linux"], "cpu": ["arm64"], "libc": ["glibc"], "what": "Linux on arm64 (glibc 2.28 or later)"},
}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def pack(folder: Path, out: Path) -> Path:
    """`npm pack` a staged folder into out; return the tarball, under out as given."""
    # npm runs inside the staged folder, so a relative out would name a folder
    # under it: it gets the absolute path. publish.yml passes `out`.
    res = subprocess.run(
        ["npm", "pack", "--json", "--ignore-scripts", "--pack-destination", str(out.resolve())],
        cwd=folder, check=True, stdout=subprocess.PIPE, text=True,
    )
    name = json.loads(res.stdout)[0]["filename"]
    path = out / name
    if not path.is_file():
        raise release.ReleaseError(f"npm pack reported {name}, but {path} is not there")
    return path


def platform_package(tarball: Path, stage: Path, version: str) -> dict:
    """Stage one platform package; return its manifest."""
    r = release.stage(tarball, stage / "release", version=version, installed_by=INSTALLED_BY)
    t = TARGETS[r.target]
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(r.root / name, stage / name)
    (stage / "README.md").write_text(
        f"# {t['name']}\n\nThe superfluid {version} release for {t['what']}. "
        f"[`{MAIN}`](https://www.npmjs.com/package/{MAIN}) installs it and runs it; depend on that, not on this.\n"
    )
    manifest = {
        "name": t["name"],
        "version": version,
        "description": f"superfluid for {t['what']}",
        "license": "Apache-2.0",
        "repository": REPOSITORY,
        "homepage": HOMEPAGE,
        "os": t["os"],
        "cpu": t["cpu"],
        **({"libc": t["libc"]} if "libc" in t else {}),
        "files": ["release/", "LICENSE", "NOTICE", "README.md"],
        "preferUnplugged": True,
    }
    write_json(stage / "package.json", manifest)
    return manifest


def main_package(stage: Path, version: str, legal: Path) -> None:
    """Stage the main package, whose commands run the platform package's binaries."""
    (stage / "bin").mkdir(parents=True)
    (stage / "lib").mkdir()
    shutil.copyfile(HERE / "launch.js", stage / "lib" / "launch.js")
    bins = {}
    for command in release.COMMANDS:
        script = stage / "bin" / f"{command}.js"
        script.write_text(f'#!/usr/bin/env node\n"use strict";\nrequire("../lib/launch.js")("{command}");\n')
        script.chmod(0o755)
        bins[command] = f"bin/{command}.js"
    shutil.copyfile(HERE / "README.md", stage / "README.md")
    for name in ("LICENSE", "NOTICE"):
        shutil.copyfile(legal / name, stage / name)
    write_json(stage / "package.json", {
        "name": MAIN,
        "version": version,
        "description": DESCRIPTION,
        "license": "Apache-2.0",
        "repository": REPOSITORY,
        "homepage": HOMEPAGE,
        "bugs": "https://github.com/basecompute/superfluid/issues",
        "keywords": ["llm", "inference", "serving", "llama.cpp", "mlx", "openai", "anthropic", "ollama"],
        "bin": bins,
        "files": ["bin/", "lib/", "LICENSE", "NOTICE", "README.md"],
        "engines": {"node": ">=18"},
        "os": ["darwin", "linux"],
        "cpu": ["arm64", "x64"],
        "optionalDependencies": {t["name"]: version for t in TARGETS.values()},
    })


def build(assets: Path, out: Path, version: str) -> list[Path]:
    """Write the four npm tarballs for a release; return them, the main package last."""
    out.mkdir(parents=True, exist_ok=True)
    built = []
    with tempfile.TemporaryDirectory() as tmp:
        stages = Path(tmp)
        for target, t in TARGETS.items():
            tarball = assets / f"superfluid-{target}.tar.gz"
            if not tarball.is_file():
                raise release.ReleaseError(f"no {tarball.name} in {assets}")
            stage = stages / t["name"].split("/")[1]
            platform_package(tarball, stage, version)
            built.append(pack(stage, out))
        main = stages / "superfluid"
        main_package(main, version, stages / TARGETS["aarch64-apple-darwin"]["name"].split("/")[1])
        built.append(pack(main, out))
    return built


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("assets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--version", required=True, help="the release version, without the v")
    a = p.parse_args()
    try:
        for path in build(a.assets, a.out, a.version):
            print(path)
    except release.ReleaseError as e:
        sys.exit(f"build_packages.py: {e}")


if __name__ == "__main__":
    main()
