"""The commands `pip install superfluid-cli` puts on PATH.

packaging/python/build_wheel.py puts the release tree beside this file, in
release/. Each command replaces this Python process with the matching binary
from it (execv, not a child), so signals, the exit status and the terminal
reach superfluid directly, and superfluid, seeing its real path, finds its
runtime workers in release/libexec/superfluid and the installed-by file that
tells `superfluid self update` to leave this copy to pip.
"""

import os
import sys
from pathlib import Path

__version__ = "0+unknown"
RELEASE = Path(__file__).resolve().parent / "release"
# The first bytes of a 64-bit Mach-O file and of an ELF file.
_BINARY = (b"\xcf\xfa\xed\xfe", b"\x7fELF")


def _restore_modes() -> None:
    """Mark the release's binaries executable again, found by header, for an installer that dropped the bits."""
    for d in (RELEASE / "bin", RELEASE / "libexec" / "superfluid"):
        for f in d.iterdir():
            with open(f, "rb") as fh:
                if fh.read(4) in _BINARY:
                    f.chmod(0o755)


def _exec(name: str) -> None:
    exe = RELEASE / "bin" / name
    if exe.is_file() and not os.access(exe, os.X_OK):
        try:
            _restore_modes()
        except OSError:
            pass
    if not os.access(exe, os.X_OK):
        sys.exit(f"{name}: {exe} is missing or not executable; `pip install --force-reinstall superfluid-cli` puts it back")
    os.execv(exe, [str(exe), *sys.argv[1:]])


def main() -> None:
    _exec("superfluid")


def workerd() -> None:
    _exec("superfluid-workerd")


def noded() -> None:
    _exec("superfluid-noded")
