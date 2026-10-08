#!/bin/sh
# Publish every workspace crate to crates.io, in dependency order, skipping the
# ones already there and waiting out crates.io's new-crate rate limit.
#
#   CARGO_REGISTRY_TOKEN=... scripts/publish-crates.sh
set -eu
[ -n "${CARGO_REGISTRY_TOKEN:-}" ] || { echo "CARGO_REGISTRY_TOKEN is not set; skipping"; exit 0; }

order="$(cargo metadata --format-version 1 --no-deps | python3 -c '
import json, sys
m = json.load(sys.stdin)
members = set(m["workspace_members"])
packages = {p["id"]: p for p in m["packages"] if p["id"] in members}
by_name = {p["name"]: p["id"] for p in packages.values()}
deps = {pid: [by_name[d["name"]] for d in p["dependencies"] if d["name"] in by_name and d.get("kind") in (None, "build")] for pid, p in packages.items()}
order, seen = [], set()
def visit(pid):
    if pid in seen:
        return
    seen.add(pid)
    for d in deps[pid]:
        visit(d)
    order.append(pid)
for pid in sorted(packages):
    visit(pid)
print(" ".join(packages[pid]["name"] + "@" + packages[pid]["version"] for pid in order))
')"

for crate in $order; do
  name="${crate%@*}"
  version="${crate#*@}"
  if curl -fsS -A "superfluid-release" "https://crates.io/api/v1/crates/$name/$version" >/dev/null 2>&1; then
    echo "$name $version is on crates.io already"
    continue
  fi
  attempt=0
  while :; do
    if cargo publish -p "$name" --locked --no-verify >/tmp/publish.log 2>&1; then
      cat /tmp/publish.log
      break
    fi
    cat /tmp/publish.log
    attempt=$((attempt + 1))
    if grep -q "429" /tmp/publish.log && [ "$attempt" -le 24 ]; then
      echo "crates.io rate limit; waiting 10 minutes ($attempt)"
      sleep 600
    else
      echo "publishing $name failed"
      exit 1
    fi
  done
done
echo "every crate is on crates.io"
