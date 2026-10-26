#!/bin/sh
# Install one kind of package packaging/build.py built, somewhere throwaway on
# this machine, and check that the commands it puts on PATH run the release
# inside it. Nothing is uploaded.
#
#   packaging/try-install.sh pip  <out-dir> <version>   a fresh venv (python3)
#   packaging/try-install.sh npm  <out-dir> <version>   a fresh npm --global prefix
#   packaging/try-install.sh brew <out-dir> <version>   a throwaway local tap; the formula is
#                                                       installed, tested, then uninstalled
#
# <out-dir> is build.py's: pypi/, npm/ and homebrew/. pip and npm install from
# those files alone; brew downloads the release tarball from GitHub, as a
# user's would, so the release must be published.
set -eu

[ $# -eq 3 ] || { echo "usage: $0 pip|npm|brew <out-dir> <version>" >&2; exit 2; }
kind=$1
out=$(cd "$2" && pwd)
version=$3

fail() { echo "try-install ($kind): $*" >&2; exit 1; }
tmp=$(mktemp -d)
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM
export SUPERFLUID_HOME="$tmp/home" SUPERFLUID_NO_UPDATE_CHECK=1

# check <bin-dir> <manager> <release-dir>: the commands in <bin-dir> run the
# release in <release-dir> and leave updates to <manager>.
check() {
  bin=$1 manager=$2 where=$3
  got=$("$bin/superfluid" --version)
  [ "$got" = "superfluid $version" ] || fail "superfluid --version said '$got', not 'superfluid $version'"
  "$bin/superfluid" --help | grep -q '^Usage: superfluid' || fail "superfluid --help printed no usage"
  for cmd in superfluid-workerd superfluid-noded; do
    "$bin/$cmd" --help 2>&1 | grep -q "usage: $cmd" || fail "$cmd did not start"
  done
  # Its shipped runtime workers are found beside the real binary, in the package.
  "$bin/superfluid" runtimes --json | python3 -c '
import json, os, sys
where = os.path.realpath(sys.argv[1])
shipped = {r["id"]: r["worker"]["path"] for r in json.load(sys.stdin) if (r.get("worker") or {}).get("source") == "shipped with superfluid"}
assert "llamacpp" in shipped, f"no shipped llama.cpp worker: {shipped}"
for rid, path in shipped.items():
    assert os.path.realpath(path).startswith(where + os.sep), f"{rid} worker {path} is not in {where}"
print("workers:", ", ".join(sorted(shipped)))
' "$where" || fail "the shipped workers were not found in $where"
  # 0.1.4 is the last release that does not read installed-by.
  if [ "$(printf '%s\n%s\n' "$version" 0.1.4 | sort -V | tail -n 1)" != 0.1.4 ]; then
    if said=$("$bin/superfluid" self update --dry-run 2>&1); then fail "self update did not leave the copy to $manager: $said"; fi
    echo "$said" | grep -q "was installed by $manager;" || fail "self update did not name $manager: $said"
  fi
  echo "try-install ($kind): superfluid $version runs from $where"
}

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) npm_platform=darwin-arm64 ;;
  Linux-x86_64) npm_platform=linux-x64 ;;
  Linux-aarch64 | Linux-arm64) npm_platform=linux-arm64 ;;
  *) fail "no release for $(uname -s) $(uname -m)" ;;
esac

case "$kind" in
  pip)
    python3 -m venv "$tmp/venv"
    # pip picks the wheel whose platform tag fits this machine, as from PyPI.
    "$tmp/venv/bin/pip" install --quiet --no-index --pre --find-links "$out/pypi" superfluid-cli
    site=$("$tmp/venv/bin/python" -c 'import superfluid_cli, os; print(os.path.dirname(superfluid_cli.__file__))')
    check "$tmp/venv/bin" pip "$site/release"
    [ "$("$tmp/venv/bin/python" -m superfluid_cli --version)" = "superfluid $version" ] || fail "python -m superfluid_cli did not run superfluid"
    ;;
  npm)
    # The machine's own package, then the commands, as npm would fetch them.
    npm install --global --prefix "$tmp/npm" --no-audit --no-fund \
      "$out/npm/basecompute-superfluid-$npm_platform-$version.tgz" "$out/npm/basecompute-superfluid-$version.tgz"
    check "$tmp/npm/bin" npm "$tmp/npm/lib/node_modules/@basecompute/superfluid-$npm_platform/release"
    ;;
  brew)
    command -v brew >/dev/null 2>&1 || fail "brew is not installed"
    export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ENV_HINTS=1
    if brew list --formula superfluid >/dev/null 2>&1; then fail "Homebrew has a superfluid installed already; this would replace it"; fi
    tap=superfluid-packaging/try
    brew tap-new --no-git "$tap" >/dev/null
    cleanup() { brew uninstall --formula "$tap/superfluid" >/dev/null 2>&1 || true; brew untap "$tap" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
    cp "$out/homebrew/superfluid.rb" "$(brew --repository "$tap")/Formula/superfluid.rb"
    brew style "$tap/superfluid"
    brew audit --strict --formula "$tap/superfluid"
    brew install --formula "$tap/superfluid"
    brew test "$tap/superfluid"
    check "$(brew --prefix)/bin" Homebrew "$(brew --cellar)/superfluid/$version/libexec"
    ;;
  *) fail "no such kind: $kind (pip, npm or brew)" ;;
esac
