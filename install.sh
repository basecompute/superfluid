#!/bin/sh
# Install superfluid from a GitHub release.
#
#   curl -fsSL https://superfluid.sh/install.sh | sh
#
# Environment:
#   SUPERFLUID_VERSION  a release tag (v0.1.0); default: the latest release
#   SUPERFLUID_NO_BASERT  set to skip installing the baseRT engine
#   SUPERFLUID_PREFIX   where to install; default: ~/.local (bin/ and libexec/superfluid/ under it)
#
# Nothing runs as root and nothing outside the prefix is touched. The baseRT
# engine is installed where it runs (Apple silicon, Linux arm64 with CUDA);
# llama.cpp and MLX install themselves the first time a model needs them.
set -eu

REPO="basecompute/superfluid"
PREFIX="${SUPERFLUID_PREFIX:-$HOME/.local}"
VERSION="${SUPERFLUID_VERSION:-latest}"

say() { printf 'superfluid: %s\n' "$*"; }
fail() { printf 'superfluid: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) target=aarch64-apple-darwin ;;
  Linux-x86_64) target=x86_64-unknown-linux-gnu ;;
  Linux-aarch64 | Linux-arm64) target=aarch64-unknown-linux-gnu ;;
  Darwin-x86_64) fail "Intel Macs are not supported; build from source (see docs/getting_started/installation.md)" ;;
  *) fail "no release for $(uname -s) $(uname -m); build from source (see docs/getting_started/installation.md)" ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
if command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | awk '{print $1}'; }
else
  fail "shasum or sha256sum is required to check the download"
fi

name="superfluid-$target"
if [ "$VERSION" = latest ]; then
  base="https://github.com/$REPO/releases/latest/download"
else
  base="https://github.com/$REPO/releases/download/$VERSION"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

say "downloading $name ($VERSION)"
curl -fsSL --proto '=https' --tlsv1.2 --retry 3 -o "$tmp/$name.tar.gz" "$base/$name.tar.gz" \
  || fail "could not download $base/$name.tar.gz"
curl -fsSL --proto '=https' --tlsv1.2 --retry 3 -o "$tmp/$name.tar.gz.sha256" "$base/$name.tar.gz.sha256" \
  || fail "could not download its checksum"
want="$(awk '{print $1}' "$tmp/$name.tar.gz.sha256")"
have="$(sha256 "$tmp/$name.tar.gz")"
[ "$want" = "$have" ] || fail "checksum mismatch: expected $want, got $have"

tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
mkdir -p "$PREFIX/bin" "$PREFIX/libexec/superfluid"
cp "$tmp/$name"/bin/* "$PREFIX/bin/"
cp "$tmp/$name"/libexec/superfluid/* "$PREFIX/libexec/superfluid/"
chmod +x "$PREFIX"/bin/superfluid* "$PREFIX"/libexec/superfluid/superfluid-worker-*

say "installed $("$PREFIX/bin/superfluid" --version) to ${PREFIX/#$HOME/~}/bin"

# The baseRT engine ships for Apple silicon and Linux arm64 (CUDA); install it
# where it runs so .base bundles serve without a first-use wait. llama.cpp and
# MLX install themselves the first time a model needs them.
case "$target" in
  aarch64-apple-darwin | aarch64-unknown-linux-gnu)
    if [ -z "${SUPERFLUID_NO_BASERT:-}" ]; then
            "$PREFIX/bin/superfluid" runtime install basert \
        || say "the baseRT engine was not installed; .base bundles need it (superfluid runtime install basert), GGUF and MLX models do not"
    fi
    ;;
esac
case ":$PATH:" in
  *":$PREFIX/bin:"*) ;;
  *) say "add $PREFIX/bin to your PATH, e.g.: export PATH=\"$PREFIX/bin:\$PATH\"" ;;
esac
say "get started: superfluid serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
