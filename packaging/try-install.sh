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
#
# On Windows this runs in Git Bash (pip and npm; Homebrew has no Windows).
set -eu

[ $# -eq 3 ] || { echo "usage: $0 pip|npm|brew <out-dir> <version>" >&2; exit 2; }
kind=$1
here=$(cd "$(dirname "$0")" && pwd)
out=$(cd "$2" && pwd)
version=$3

fail() { echo "try-install ($kind): $*" >&2; exit 1; }
tmp=$(mktemp -d)
windows=false
python=python3
case "$(uname -s)" in
  MINGW* | MSYS*)
    windows=true
    # Python for Windows installs no python3.
    python=python
    # Git Bash names folders its own way (/tmp/..., /d/a/...), which a Windows
    # program does not read: the same folders as Windows names them.
    tmp=$(cygpath -m "$tmp")
    out=$(cygpath -m "$out")
    here=$(cygpath -m "$here")
    ;;
esac
# What a Windows program printed, less the carriage returns Python and
# PowerShell end their lines with.
lines() { tr -d '\r'; }
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM
export SUPERFLUID_HOME="$tmp/home" SUPERFLUID_NO_UPDATE_CHECK=1 SUPERFLUID_NO_UPDATE=1

# check <bin-dir> <manager> <release-dir>: the commands in <bin-dir> run the
# release in <release-dir> and know <manager> as what updates it.
check() {
  bin=$1 manager=$2 where=$3
  got=$("$bin/superfluid" --version | lines)
  [ "$got" = "superfluid $version" ] || fail "superfluid --version said '$got', not 'superfluid $version'"
  "$bin/superfluid" --help | grep -q '^Usage: superfluid' || fail "superfluid --help printed no usage"
  # The command's status is the program's own, through whatever launcher runs it.
  if "$bin/superfluid" no-such-command >/dev/null 2>&1; then fail "superfluid no-such-command succeeded: the launcher lost the program's status"; fi
  for cmd in superfluid-workerd superfluid-noded; do
    "$bin/$cmd" --help 2>&1 | grep -q "usage: $cmd" || fail "$cmd did not start"
  done
  # Its runtime workers are the package's own: an adapter the release ships,
  # or, for a runtime linked into the daemon (the Windows release's llama.cpp),
  # the package's superfluid-workerd.
  "$bin/superfluid" runtimes --json | "$python" "$here/own_workers.py" "$where" || fail "the package's own workers were not found in $where"
  # 0.1.4 is the last release that does not read installed-by. 0.1.5 named
  # every manager's command and left the update to the user ("was installed
  # by <manager>; `<command>` updates it", exit 1). From 0.1.6 on, `self
  # update` runs the manager itself where it can — `--dry-run` then says what
  # it would run and exits 0 — and otherwise names the command, as before:
  # pip always (its environment may be a project's or the system's), npm
  # here, since a throwaway --prefix has no npm of its own to run, and every
  # manager on Windows, which lets none replace a program that is running.
  # Either way the copy is the manager's, named, and a dry run changes nothing.
  if [ "$(printf '%s\n%s\n' "$version" 0.1.4 | sort -V | tail -n 1)" != 0.1.4 ]; then
    if said=$("$bin/superfluid" self update --dry-run 2>&1); then rc=0; else rc=$?; fi
    echo "$said" | grep -q "was installed by $manager[;:,]" || fail "self update did not name $manager: $said"
    if [ "$rc" -eq 0 ]; then
      echo "$said" | grep -q 'would run' || fail "self update --dry-run exited 0 without showing $manager's command: $said"
    else
      echo "$said" | grep -q 'updates it' || fail "self update did not name $manager's command: $said"
    fi
    [ "$("$bin/superfluid" --version | lines)" = "superfluid $version" ] || fail "self update --dry-run changed the copy"
  fi
  echo "try-install ($kind): superfluid $version runs from $where"
}

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) npm_platform=darwin-arm64 ;;
  Linux-x86_64) npm_platform=linux-x64 ;;
  Linux-aarch64 | Linux-arm64) npm_platform=linux-arm64 ;;
  MINGW*-x86_64 | MSYS*-x86_64) npm_platform=win32-x64 ;;
  *) fail "no release for $(uname -s) $(uname -m)" ;;
esac

case "$kind" in
  pip)
    "$python" -m venv "$tmp/venv"
    # A venv's commands are in bin, and on Windows in Scripts.
    venv_bin="$tmp/venv/bin"
    if $windows; then venv_bin="$tmp/venv/Scripts"; fi
    # pip picks the wheel whose platform tag fits this machine, as from PyPI.
    "$venv_bin/pip" install --quiet --no-index --pre --find-links "$out/pypi" superfluid-cli
    site=$("$venv_bin/python" -c 'import superfluid_cli, os; print(os.path.dirname(superfluid_cli.__file__))' | lines)
    check "$venv_bin" pip "$site/release"
    [ "$("$venv_bin/python" -m superfluid_cli --version | lines)" = "superfluid $version" ] || fail "python -m superfluid_cli did not run superfluid"
    ;;
  npm)
    # The machine's own package, then the commands, as npm would fetch them.
    npm install --global --prefix "$tmp/npm" --no-audit --no-fund \
      "$out/npm/basecompute-superfluid-$npm_platform-$version.tgz" "$out/npm/basecompute-superfluid-$version.tgz"
    if $windows; then
      # npm on Windows has no bin or lib under its prefix: the commands are in
      # the prefix itself, as a script for each shell. Git Bash runs the sh
      # one in check; cmd and PowerShell users get superfluid.cmd.
      check "$tmp/npm" npm "$tmp/npm/node_modules/@basecompute/superfluid-$npm_platform/release"
      got=$(powershell -NoProfile -Command "& '$tmp/npm/superfluid.cmd' --version" | lines)
      [ "$got" = "superfluid $version" ] || fail "superfluid.cmd --version said '$got', not 'superfluid $version'"
    else
      check "$tmp/npm/bin" npm "$tmp/npm/lib/node_modules/@basecompute/superfluid-$npm_platform/release"
    fi
    ;;
  brew)
    command -v brew >/dev/null 2>&1 || fail "brew is not installed"
    export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ENV_HINTS=1
    if brew list --formula superfluid >/dev/null 2>&1; then fail "Homebrew has a superfluid installed already; this would replace it"; fi
    tap=superfluid-packaging/try
    brew tap-new --no-git "$tap" >/dev/null
    # Homebrew 6 and later load a non-official tap's formula only once the tap
    # is trusted; an older Homebrew has no such command and needs none.
    brew trust --tap "$tap" >/dev/null 2>&1 || true
    cleanup() { brew uninstall --formula "$tap/superfluid" >/dev/null 2>&1 || true; brew untap "$tap" >/dev/null 2>&1 || true; brew untrust --tap "$tap" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
    cp "$out/homebrew/superfluid.rb" "$(brew --repository "$tap")/Formula/superfluid.rb"
    brew style "$tap/superfluid"
    brew audit --strict --formula "$tap/superfluid"
    brew install --formula "$tap/superfluid"
    brew test "$tap/superfluid"
    check "$(brew --prefix)/bin" Homebrew "$(brew --cellar)/superfluid/$version/libexec"
    ;;
  *) fail "no such kind: $kind (pip, npm or brew)" ;;
esac
