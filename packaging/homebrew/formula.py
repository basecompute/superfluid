"""Write the Homebrew formula (superfluid.rb) for one superfluid release.

    python3 packaging/homebrew/formula.py <assets-dir> <out-dir> --version <v>

<assets-dir> holds the release's three tarballs and their .sha256 files. The
formula downloads the same tarballs from the GitHub release, so it takes their
checksums from the .sha256 files, after checking each against its tarball, and
the oldest macOS it allows from the macOS binaries themselves. It is written
whole from the template below, so from one release to the next only the
version, the checksums and (rarely) the macOS line change, and the PR the
publish workflow opens on basecompute/homebrew-tap shows exactly that.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import release  # noqa: E402

REPO = "basecompute/superfluid"
INSTALLED_BY = "Homebrew: brew upgrade superfluid"
# Homebrew's names for the macOS versions a release could require.
MACOS = {13: "ventura", 14: "sonoma", 15: "sequoia", 26: "tahoe"}

TEMPLATE = """\
# Written by superfluid's publish workflow (packaging/homebrew/formula.py in
# basecompute/superfluid) for each release; a change made here is overwritten.
class Superfluid < Formula
  desc "Serving daemon for local LLM inference across runtimes and machines"
  homepage "https://superfluid.sh"
  license "Apache-2.0"

  # One prebuilt release per platform; Homebrew reads the version from the url.
  # (Each url sits under a CPU check: Homebrew's style rules allow none directly
  # in on_macos / on_linux.)
  on_macos do
    depends_on arch: :arm64
    depends_on macos: :{macos}
    if Hardware::CPU.arm?
      url "{base}/superfluid-aarch64-apple-darwin.tar.gz"
      sha256 "{sha_darwin}"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "{base}/superfluid-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "{sha_linux_x64}"
    end
    if Hardware::CPU.arm?
      url "{base}/superfluid-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "{sha_linux_arm64}"
    end
  end

  def install
    # The release keeps its layout in libexec: superfluid finds its runtime
    # workers in libexec/superfluid beside its own bin/, through the links below.
    libexec.install "bin", "libexec", "compat.json"
    # Tells `superfluid self update` that Homebrew owns this copy.
    (libexec/"installed-by").write "{installed_by}\\n"
    bin.install_symlink libexec.glob("bin/*")
  end

  test do
    assert_match "superfluid #{{version}}", shell_output("#{{bin}}/superfluid --version")
    # The shipped runtime workers are found beside the real binary.
    assert_match libexec.to_s, shell_output("#{{bin}}/superfluid runtimes --json")
  end
end
"""


def checksum(assets: Path, target: str) -> str:
    tarball = assets / f"superfluid-{target}.tar.gz"
    if not tarball.is_file():
        raise release.ReleaseError(f"no {tarball.name} in {assets}")
    release.check_sha256(tarball)
    return (assets / f"{tarball.name}.sha256").read_text().split()[0]


def render(version: str, shas: dict[str, str], macos: tuple[int, int]) -> str:
    # Homebrew names only whole macOS versions, so 15.4 means the next one, never 15.0.
    named = [major for major in sorted(MACOS) if (major, 0) >= macos]
    if not named:
        raise release.ReleaseError(f"no Homebrew name for a macOS that has {macos[0]}.{macos[1]}; add it to MACOS")
    return TEMPLATE.format(
        version=version,
        base=f"https://github.com/{REPO}/releases/download/v{version}",
        macos=MACOS[named[0]],
        sha_darwin=shas["aarch64-apple-darwin"],
        sha_linux_x64=shas["x86_64-unknown-linux-gnu"],
        sha_linux_arm64=shas["aarch64-unknown-linux-gnu"],
        installed_by=INSTALLED_BY,
    )


def build(assets: Path, out: Path, version: str) -> Path:
    shas = {target: checksum(assets, target) for target in release.TARGETS}
    with tempfile.TemporaryDirectory() as tmp:
        r = release.stage(assets / "superfluid-aarch64-apple-darwin.tar.gz", Path(tmp) / "release", version=version)
        macos = release.macos_requirement(r.root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "superfluid.rb"
    path.write_text(render(version, shas, macos))
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("assets", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--version", required=True, help="the release version, without the v")
    a = p.parse_args()
    try:
        print(build(a.assets, a.out, a.version))
    except release.ReleaseError as e:
        sys.exit(f"formula.py: {e}")


if __name__ == "__main__":
    main()
