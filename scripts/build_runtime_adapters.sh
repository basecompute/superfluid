#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?usage: build_runtime_adapters.sh <stage-dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/Cargo.toml"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}/release"
OS="$(uname -s)"
ARCH="$(uname -m)"

say() { echo "[adapters] $*"; }
fail() { echo "::error::[adapters] $*"; exit 1; }

command -v cargo >/dev/null 2>&1 || export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null 2>&1 || fail "cargo not found"

case "$OS" in
  Darwin) TOKENIZER=libsuperfluid_tokenizer_llamacpp.dylib
          export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}" ;;
  Linux)  TOKENIZER=libsuperfluid_tokenizer_llamacpp.so ;;
  *)      fail "no adapters are built for $OS" ;;
esac
MLX=""
[ "$OS" = Darwin ] && [ "$ARCH" = arm64 ] && MLX=1

say "building the llama.cpp adapter (worker + tokenizer library)"
cargo build --release --manifest-path "$MANIFEST" \
  -p superfluid-adapter-llamacpp -p superfluid-tokenizer-llamacpp

# The baseRT worker installs the engine (its release ships a worker of its own
# that serves); the engine runs on Apple silicon and Linux arm64.
BASERT=""
{ [ "$OS" = Darwin ] && [ "$ARCH" = arm64 ]; } || { [ "$OS" = Linux ] && [ "$ARCH" = aarch64 ]; } && BASERT=1
say "building the baseRT adapter's worker"
cargo build --release --manifest-path "$MANIFEST" -p superfluid-adapter-basert --bin superfluid-worker-basert

if [ -n "$MLX" ]; then
  RECIPE="$ROOT/crates/superfluid-adapters/superfluid-adapter-mlx/recipe.json"
  PY_FIELDS="$(python3 - "$RECIPE" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["python"]["assets"]["macos-arm64"]
print(a["name"]); print(a["url"]); print(a["sha256"])
PY
)"
  PY_NAME="$(printf '%s\n' "$PY_FIELDS" | sed -n 1p)"
  PY_URL="$(printf '%s\n' "$PY_FIELDS" | sed -n 2p)"
  PY_SHA="$(printf '%s\n' "$PY_FIELDS" | sed -n 3p)"
  [ -n "$PY_NAME" ] && [ -n "$PY_URL" ] && [ -n "$PY_SHA" ] || fail "recipe.json names no Python for macos-arm64"

  if [ -n "${SUPERFLUID_ADAPTER_CACHE:-}" ]; then
    CACHE="$SUPERFLUID_ADAPTER_CACHE"
  else
    CACHE="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/superfluid-adapter-python"
  fi
  mkdir -p "$CACHE"
  ARCHIVE="${SUPERFLUID_ADAPTER_PYTHON_ARCHIVE:-$CACHE/$PY_NAME}"
  if [ ! -f "$ARCHIVE" ]; then
    say "fetching $PY_URL"
    curl -fSL --proto '=https' --tlsv1.2 --retry 3 -o "$ARCHIVE.part" "$PY_URL" \
      || { rm -f "$ARCHIVE.part"; fail "could not fetch $PY_URL"; }
    mv "$ARCHIVE.part" "$ARCHIVE"
  fi
  HAVE_SHA="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
  [ "$HAVE_SHA" = "$PY_SHA" ] \
    || fail "$ARCHIVE has SHA-256 $HAVE_SHA, and the recipe names $PY_SHA: not the Python an install fetches"
  PYDIR="$CACHE/${PY_NAME%.tar.gz}"
  if [ ! -f "$PYDIR/.unpacked" ]; then
    rm -rf "$PYDIR" "$PYDIR.part"; mkdir -p "$PYDIR.part"
    tar -xzf "$ARCHIVE" -C "$PYDIR.part"
    touch "$PYDIR.part/.unpacked"
    mv "$PYDIR.part" "$PYDIR"
  fi
  [ -x "$PYDIR/python/bin/python3" ] || fail "$PY_NAME holds no python/bin/python3"

  say "building the MLX adapter's worker against $PY_NAME"
  PYO3_PYTHON="$PYDIR/python/bin/python3" \
    cargo build --release --manifest-path "$MANIFEST" -p superfluid-adapter-mlx --bin superfluid-worker-mlx
fi

rm -rf "$STAGE"; mkdir -p "$STAGE"
cp "$TARGET/superfluid-worker-llamacpp" "$TARGET/$TOKENIZER" "$TARGET/superfluid-worker-basert" "$STAGE/"
[ -n "$MLX" ] && cp "$TARGET/superfluid-worker-mlx" "$STAGE/"
chmod +x "$STAGE"/superfluid-worker-*

if [ "$OS" = Darwin ]; then
  for f in "$STAGE"/*; do
    minos="$(otool -l "$f" | awk '/LC_BUILD_VERSION/{b=1} b && /minos/{print $2; exit}')"
    [ "$minos" = "$MACOSX_DEPLOYMENT_TARGET" ] \
      || fail "$(basename "$f") minos is '$minos' (want $MACOSX_DEPLOYMENT_TARGET): macOS $MACOSX_DEPLOYMENT_TARGET hosts could not run it"
  done
  nm -gU "$STAGE/$TOKENIZER" | grep -q ' _superfluid_tokenizer_open$' \
    || fail "$TOKENIZER does not export the tokenizer interface"
else
  nm -D --defined-only "$STAGE/$TOKENIZER" | grep -q ' superfluid_tokenizer_open$' \
    || fail "$TOKENIZER does not export the tokenizer interface"
fi

if [ -n "$MLX" ]; then
  W="$STAGE/superfluid-worker-mlx"
  PY_LINK="$(otool -l "$W" | awk '/cmd LC_LOAD(_WEAK)?_DYLIB/{c=$2} /^ *name /{if ($2 ~ /libpython3|\/Python$/) print c, $2}')"
  case "$PY_LINK" in
    "LC_LOAD_WEAK_DYLIB @rpath/libpython3."*".dylib") : ;;
    *) fail "the MLX worker's Python link is '$PY_LINK' (want a weak @rpath/libpython3.x.dylib): it would not start, or not find an install's Python" ;;
  esac
  otool -l "$W" | grep -A2 'cmd LC_RPATH' | grep -q 'path @executable_path/../python/lib ' \
    || fail "the MLX worker has no @executable_path/../python/lib rpath: it would not find an install's Python"
  if otool -l "$W" | grep -A2 'cmd LC_RPATH' | grep 'path ' | grep -qv '@executable_path'; then
    fail "the MLX worker has an rpath into this machine: $(otool -l "$W" | grep -A2 'cmd LC_RPATH' | grep 'path ')"
  fi
fi

EMPTY="$(mktemp -d)"
trap 'rm -rf "$EMPTY"' EXIT
bare() { env -i HOME="$EMPTY" SUPERFLUID_HOME="$EMPTY/superfluid" PATH=/usr/bin:/bin "$@"; }

# check_adapter <id> [starts]: the worker starts and reports with no runtime
# here, and plans its install; "starts" asks for the report only, where a plan
# cannot be answered on the builder (the baseRT engine for Linux arm64 wants an
# NVIDIA GPU the builder lacks).
check_adapter() {
  local id="$1" worker="$STAGE/superfluid-worker-$1" report plans
  if report="$(bare "$worker" check 2>"$EMPTY/check.err")"; then
    fail "$id: check passed with no runtime installed"
  fi
  printf '%s' "$report" | python3 -c '
import json, sys
id = sys.argv[1]
try:
    r = json.load(sys.stdin)
except Exception as e:
    sys.exit(f"no report: {e}")
why = r.get("available")
assert r["runtime"]["id"] == id, r["runtime"]
assert isinstance(why, str) and f"superfluid runtime install {id}" in why, f"available = {why!r}"
assert r.get("formats"), "it declares no format"
print(f"[adapters] {id}: starts with no runtime here, and says: {why}")
' "$id" || { cat "$EMPTY/check.err" >&2; fail "$id: the worker does not start and report with no runtime installed (above)"; }
  [ "${2:-}" = starts ] && return 0
  plans="$(bare "$worker" plan 2>"$EMPTY/plan.err")" \
    || { cat "$EMPTY/plan.err" >&2; fail "$id: the worker cannot plan its install"; }
  printf '%s' "$plans" | python3 -c '
import json, sys
id = sys.argv[1]
p = json.load(sys.stdin)["plans"][0]
assert p["id"] == id and p["tested"] and p["assets"], p
assert all(a.get("sha256") for a in p["assets"]), "an asset with no SHA-256"
install, names = p["install"], ", ".join(a["name"] for a in p["assets"])
print(f"[adapters] {id}: plans {install}: {names}")
' "$id" || fail "$id: its plan is not one an install can carry out"
}

check_adapter llamacpp
if [ -n "$BASERT" ]; then
  if [ "$OS" = Darwin ]; then check_adapter basert; else check_adapter basert starts; fi
fi
if [ -n "$MLX" ]; then
  check_adapter mlx
  bare "$STAGE/superfluid-worker-mlx" plan | grep -q "\"name\":\"$PY_NAME\"" \
    || fail "the MLX worker was not built against the Python its install fetches ($PY_NAME)"
fi

say "staged in $STAGE:"
ls -l "$STAGE" | sed 's/^/[adapters]   /'
