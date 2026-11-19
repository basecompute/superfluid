"""Build every package of one superfluid release from its GitHub release assets.

    python3 packaging/build.py <assets-dir> <out-dir> --version <v> [--only pypi,npm,homebrew] [--targets <target>,...]

<assets-dir> holds the release's tarballs and their .sha256 files
(`gh release download v<v> --pattern 'superfluid-*.tar.gz*'`). Writes:

    <out-dir>/pypi/superfluid_cli-<v>-py3-none-<platform>.whl   one per platform        (python/build_wheel.py)
    <out-dir>/npm/basecompute-superfluid[-<platform>]-<v>.tgz    one per platform, and
                                                                 the commands            (npm/build_packages.py)
    <out-dir>/homebrew/superfluid.rb                             the tap's formula       (homebrew/formula.py)

Every platform is packaged (TARGETS in release.py), less one a release may
come without (OPTIONAL there) when its tarball is absent. --targets names the
platforms instead, for trying one platform's wheel and npm packages where its
tarball was just built (release.yml does, on Windows); the formula always
needs its own three.

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


def build(assets: Path, out: Path, version: str, kinds=KINDS, targets: list[str] | None = None) -> list[Path]:
    targets = release.targets_in(assets) if targets is None else targets
    built = []
    if "pypi" in kinds:
        for target in targets:
            built.append(build_wheel.build(assets / f"superfluid-{target}.tar.gz", out / "pypi", version))
    if "npm" in kinds:
        built += build_packages.build(assets, out / "npm", version, targets)
    if "homebrew" in kinds:
        built.append(formula.build(assets, out / "homebrew", version))
    return built


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("assets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--version", required=True, help="the release version, without the v")
    p.add_argument("--only", default=",".join(KINDS), help=f"a comma-separated subset of {', '.join(KINDS)}")
    p.add_argument("--targets", help=f"package these platforms alone: a comma-separated subset of {', '.join(release.TARGETS)}")
    a = p.parse_args()
    kinds = a.only.split(",")
    unknown = set(kinds) - set(KINDS)
    if unknown:
        p.error(f"--only: no such kind {', '.join(sorted(unknown))}")
    targets = a.targets.split(",") if a.targets else None
    unknown = set(targets or ()) - set(release.TARGETS)
    if unknown:
        p.error(f"--targets: no such target {', '.join(sorted(unknown))}")
    try:
        for path in build(a.assets, a.out, a.version.lstrip("v"), kinds, targets):
            print(path)
    except release.ReleaseError as e:
        sys.exit(f"build.py: {e}")


if __name__ == "__main__":
    main()
