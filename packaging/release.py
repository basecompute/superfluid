"""Read a superfluid release tarball and lay it out for a package manager.

    python3 packaging/release.py <superfluid-<target>.tar.gz> <dest-dir> [--version <v>] [--installed-by "<manager>: <command>"]

Every package (PyPI, npm, Homebrew) carries the release tree exactly as the
GitHub release has it: bin/ with the commands, libexec/superfluid/ with the
runtime workers superfluid finds beside its own bin/, compat.json, LICENSE and
NOTICE. Staging changes two things:

* it writes an `installed-by` file at the root, `<manager>: <command>`, which
  tells `superfluid self update`, and the line `serve` prints when a newer
  release is out, that a package manager owns this copy and how it is updated;
* it sets each file's mode by what the file is, 0755 for a Mach-O or ELF
  binary and 0644 for the rest, since neither a wheel nor an npm tarball is
  trusted to carry modes through.

The tarball is checked against the .sha256 beside it and read member by
member: only regular files and directories under its one top folder are
accepted, so nothing it holds lands outside the destination, and a symlink,
which neither a wheel nor an npm tarball can carry, stops the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Each platform a release ships, by the target in its asset name, and what the
# binaries for it must be.
TARGETS = {
    "aarch64-apple-darwin": {"format": "macho", "arch": "arm64"},
    "x86_64-unknown-linux-gnu": {"format": "elf", "arch": "x86_64"},
    "aarch64-unknown-linux-gnu": {"format": "elf", "arch": "aarch64"},
}
# The commands a release puts on PATH: LINKED in crates/superfluid-daemon/src/selfupdate.rs.
COMMANDS = ("superfluid", "superfluid-workerd", "superfluid-noded")
# INSTALLED_BY in crates/superfluid-daemon/src/selfupdate.rs.
INSTALLED_BY = "installed-by"

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
MACHO_CPU = {0x0100000C: "arm64", 0x01000007: "x86_64"}
LC_BUILD_VERSION = 0x32
LC_VERSION_MIN_MACOSX = 0x24
ELF_MAGIC = b"\x7fELF"
ELF_MACHINE = {62: "x86_64", 183: "aarch64"}
GLIBC_RE = re.compile(rb"GLIBC_2\.(\d+)(?:\.\d+)?\x00")


class ReleaseError(Exception):
    pass


@dataclass
class Release:
    root: Path
    target: str
    version: str


def target_of(tarball: Path) -> str:
    """The target in a release asset's name, superfluid-<target>.tar.gz."""
    m = re.fullmatch(r"superfluid-(.+)\.tar\.gz", tarball.name)
    if not m or m.group(1) not in TARGETS:
        raise ReleaseError(f"{tarball.name} is not a release tarball (superfluid-<target>.tar.gz, target one of {', '.join(TARGETS)})")
    return m.group(1)


def binary_kind(data: bytes) -> tuple[str, str] | None:
    """('macho' or 'elf', architecture) for an executable or library, read from its header; None for anything else."""
    if len(data) >= 8 and struct.unpack_from("<I", data)[0] == MH_MAGIC_64:
        cpu = struct.unpack_from("<I", data, 4)[0]
        return "macho", MACHO_CPU.get(cpu, hex(cpu))
    if len(data) >= 8 and struct.unpack_from(">I", data)[0] == FAT_MAGIC:
        return "macho", "universal"
    if len(data) >= 20 and data[:4] == ELF_MAGIC:
        machine = struct.unpack_from("<H" if data[5] == 1 else ">H", data, 18)[0]
        return "elf", ELF_MACHINE.get(machine, str(machine))
    return None


def macos_minimum(data: bytes) -> tuple[int, int] | None:
    """The oldest macOS a 64-bit Mach-O file loads on, from its LC_BUILD_VERSION (or older LC_VERSION_MIN_MACOSX)."""
    if len(data) < 32 or struct.unpack_from("<I", data)[0] != MH_MAGIC_64:
        return None
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    for _ in range(ncmds):
        if off + 16 > len(data):
            return None
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == LC_BUILD_VERSION:
            v = struct.unpack_from("<I", data, off + 12)[0]
            return v >> 16, (v >> 8) & 0xFF
        if cmd == LC_VERSION_MIN_MACOSX:
            v = struct.unpack_from("<I", data, off + 8)[0]
            return v >> 16, (v >> 8) & 0xFF
        if size < 8:
            return None
        off += size
    return None


def glibc_minimum(data: bytes) -> int | None:
    """The newest glibc symbol version an ELF file asks for, as the minor of 2.N."""
    found = [int(m) for m in GLIBC_RE.findall(data)]
    return max(found) if found else None


def binaries(root: Path) -> list[Path]:
    """Every Mach-O or ELF file in a release tree."""
    out = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and not f.is_symlink():
            with open(f, "rb") as fh:
                if binary_kind(fh.read(64)) is not None:
                    out.append(f)
    return out


def check_sha256(tarball: Path) -> None:
    sums = tarball.with_name(tarball.name + ".sha256")
    if not sums.is_file():
        raise ReleaseError(f"no {sums.name} beside {tarball.name}")
    want = (sums.read_text().split() or [""])[0]
    have = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if want != have:
        raise ReleaseError(f"{tarball.name}: SHA-256 is {have}, {sums.name} says {want or 'nothing'}")


def unpack(tarball: Path, dest: Path) -> None:
    """Extract the tarball's one top folder into dest (which must be empty or absent), accepting files and folders only."""
    if dest.exists() and any(dest.iterdir()):
        raise ReleaseError(f"{dest} is not empty")
    top = f"superfluid-{target_of(tarball)}"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        for m in tar:
            path = PurePosixPath(m.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != top:
                raise ReleaseError(f"{tarball.name}: {m.name!r} is outside {top}/")
            rel = PurePosixPath(*path.parts[1:])
            out = dest.joinpath(*rel.parts) if rel.parts else dest
            if m.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif m.isreg():
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                with open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh)
            else:
                kind = "a symlink" if m.issym() else "a hard link" if m.islnk() else "not a regular file"
                raise ReleaseError(f"{tarball.name}: {m.name} is {kind}; a wheel or npm tarball cannot carry it")


def set_modes(root: Path) -> None:
    """0755 for folders and for Mach-O or ELF files, found by header rather than by name; 0644 for every other file."""
    executables = set(binaries(root))
    for p in [root, *root.rglob("*")]:
        p.chmod(0o755 if p.is_dir() or p in executables else 0o644)


def check(root: Path, target: str, version: str | None) -> str:
    """Check that a release tree is complete and built for `target`; return its version (from compat.json)."""
    want = TARGETS[target]
    compat = root / "compat.json"
    if not compat.is_file():
        raise ReleaseError(f"no compat.json in the {target} release")
    found = json.loads(compat.read_text()).get("superfluid")
    if not isinstance(found, str) or not found:
        raise ReleaseError(f"{target}: compat.json names no superfluid version")
    if version is not None and found != version:
        raise ReleaseError(f"{target}: the release is superfluid {found}, not {version}")
    for name in ("LICENSE", "NOTICE"):
        if not (root / name).is_file():
            raise ReleaseError(f"{target}: no {name} in the release")
    for name in COMMANDS:
        if not (root / "bin" / name).is_file():
            raise ReleaseError(f"{target}: no bin/{name} in the release")
    workers = root / "libexec" / "superfluid"
    if not workers.is_dir() or not any(workers.glob("superfluid-worker-*")):
        raise ReleaseError(f"{target}: no runtime workers in libexec/superfluid")
    found_binaries = binaries(root)
    for name in COMMANDS:
        if root / "bin" / name not in found_binaries:
            raise ReleaseError(f"{target}: bin/{name} is not an executable")
    for f in found_binaries:
        with open(f, "rb") as fh:
            kind = binary_kind(fh.read(64))
        if kind != (want["format"], want["arch"]):
            raise ReleaseError(f"{target}: {f.relative_to(root)} is {kind[0]} {kind[1]}, not {want['format']} {want['arch']}")
    if (root / INSTALLED_BY).exists():
        raise ReleaseError(f"{target}: the release already has an {INSTALLED_BY} file")
    return found


def stage(tarball: Path, dest: Path, version: str | None = None, installed_by: str | None = None) -> Release:
    """Check and unpack a release tarball into dest, the release root, ready to package."""
    target = target_of(tarball)
    check_sha256(tarball)
    unpack(tarball, dest)
    found = check(dest, target, version)
    if installed_by is not None:
        manager, sep, command = installed_by.partition(": ")
        if not sep or not manager.strip() or not command.strip() or "\n" in installed_by:
            raise ReleaseError(f"--installed-by must be one line, '<manager>: <command>', not {installed_by!r}")
        (dest / INSTALLED_BY).write_text(installed_by + "\n")
    set_modes(dest)
    return Release(dest, target, found)


def macos_requirement(root: Path) -> tuple[int, int]:
    """The oldest macOS every binary in a release tree loads on."""
    found = []
    for f in binaries(root):
        v = macos_minimum(f.read_bytes())
        if v is not None:
            found.append(v)
    if not found:
        raise ReleaseError(f"{root}: no Mach-O binary states a minimum macOS")
    return max(found)


def wheel_platform(root: Path) -> str:
    """The wheel platform tag a release tree needs, read from its binaries so pip refuses it where they cannot run.

    macOS: the newest minimum any Mach-O file states. From macOS 11 on pip only
    knows X_0 tags, so a minimum of 15.4 becomes macosx_16_0 rather than
    macosx_15_0, which would let it onto 15.0-15.3. Linux: manylinux_2_N, N the
    newest glibc symbol version any ELF file needs (2.28 for the release builds).
    """
    kinds = set()
    glibc = []
    for f in binaries(root):
        data = f.read_bytes()
        kind = binary_kind(data[:64])
        assert kind is not None
        kinds.add(kind)
        if kind[0] == "elf":
            v = glibc_minimum(data)
            if v is not None:
                glibc.append(v)
    if len(kinds) != 1:
        raise ReleaseError(f"{root}: binaries of more than one kind: {sorted(kinds)}")
    (fmt, arch), = kinds
    if fmt == "macho":
        major, minor = macos_requirement(root)
        if major >= 11 and minor > 0:
            major, minor = major + 1, 0
        return f"macosx_{major}_{minor}_{arch}"
    if not glibc:
        raise ReleaseError(f"{root}: no ELF binary names a glibc version")
    return f"manylinux_2_{max(glibc)}_{arch}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tarball", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--version")
    p.add_argument("--installed-by")
    a = p.parse_args()
    try:
        r = stage(a.tarball, a.dest, a.version, a.installed_by)
        print(f"{r.root}: superfluid {r.version} for {r.target} ({wheel_platform(r.root)})")
    except ReleaseError as e:
        sys.exit(f"release.py: {e}")


if __name__ == "__main__":
    main()
