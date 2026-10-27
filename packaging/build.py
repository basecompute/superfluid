"""Build every package of one superfluid release from its GitHub release assets.

    python3 packaging/build.py <assets-dir> <out-dir> --version <v> [--only pypi,npm,homebrew]

<assets-dir> holds the release's three tarballs and their .sha256 files
(`gh release download v<v> --pattern 'superfluid-*.tar.gz*'`). Writes:

    <out-dir>/pypi/superfluid_cli-<v>-py3-none-<platform>.whl   one per platform  (python/build_wheel.py)
    <out-dir>/npm/basecompute-superfluid[-<platform>]-<v>.tgz    four               (npm/build_packages.py)
    <out-dir>/homebrew/superfluid.rb                             the tap's formula  (homebrew/formula.py)

Nothing is compiled: each package repackages the release as published, and
packaging/try-install.sh installs one kind on this machine to check it.
.github/workflows/publish.yml runs both and uploads what passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "python"))
sys.path.insert(0, str(HERE / "npm"))
sys.path.insert(0, str(HERE / "homebrew"))
import build_packages  # noqa: E402
import build_wheel  # noqa: E402
import formula  # noqa: E402
import release  # noqa: E402

KINDS = ("pypi", "npm", "homebrew")


def build(assets: Path, out: Path, version: str, kinds=KINDS) -> list[Path]:
    built = []
    if "pypi" in kinds:
        for target in release.TARGETS:
            built.append(build_wheel.build(assets / f"superfluid-{target}.tar.gz", out / "pypi", version))
    if "npm" in kinds:
        built += build_packages.build(assets, out / "npm", version)
    if "homebrew" in kinds:
        built.append(formula.build(assets, out / "homebrew", version))
    return built


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("assets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--version", required=True, help="the release version, without the v")
    p.add_argument("--only", default=",".join(KINDS), help=f"a comma-separated subset of {', '.join(KINDS)}")
    a = p.parse_args()
    kinds = a.only.split(",")
    unknown = set(kinds) - set(KINDS)
    if unknown:
        p.error(f"--only: no such kind {', '.join(sorted(unknown))}")
    try:
        for path in build(a.assets, a.out, a.version.lstrip("v"), kinds):
            print(path)
    except release.ReleaseError as e:
        sys.exit(f"build.py: {e}")


if __name__ == "__main__":
    main()
