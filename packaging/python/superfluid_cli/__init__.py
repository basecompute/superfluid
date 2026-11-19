"""The commands `pip install superfluid-cli` puts on PATH.

packaging/python/build_wheel.py puts the release tree beside this file, in
release/. Each command replaces this Python process with the matching binary
from it (execv, not a child), so signals, the exit status and the terminal
reach superfluid directly, and superfluid, seeing its real path, finds its
runtime workers in release/libexec/superfluid and the installed-by file that
tells `superfluid self update` to leave this copy to pip.

Windows has no exec: os.execv there starts the program and ends this process
at once, so the shell would prompt again while superfluid still ran, and
would never see its status. There the command runs the program as a child,
waits for it and exits with its status.
"""

import os
import sys
from pathlib import Path

__version__ = "0+unknown"
RELEASE = Path(__file__).resolve().parent / "release"
WINDOWS = os.name == "nt"
# The first bytes of a 64-bit Mach-O file and of an ELF file.
_BINARY = (b"\xcf\xfa\xed\xfe", b"\x7fELF")


def _restore_modes() -> None:
    """Mark the release's binaries executable again, found by header, for an installer that dropped the bits."""
    for d in (RELEASE / "bin", RELEASE / "libexec" / "superfluid"):
        for f in d.iterdir():
            with open(f, "rb") as fh:
                if fh.read(4) in _BINARY:
                    f.chmod(0o755)


def _wait_for(name: str, exe: Path) -> None:
    """Run `exe` with this command's arguments, console and streams, and exit with its status."""
    import signal
    import subprocess

    # Ctrl+C and Ctrl+Break reach every program on the console: superfluid
    # answers them (a server drains), and this only waits for it. A handler
    # that does nothing, not SIG_IGN: what is ignored here a child may inherit.
    for sig in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), lambda *_: None)
    try:
        status = subprocess.call([str(exe), *sys.argv[1:]])
    except OSError as e:
        sys.exit(f"{name}: could not start {exe}: {e}")
    # A Windows status is 32 bits without a sign (0xC000013A when Ctrl+C ended
    # the program) and Python exits with a C int: the same bits, as one.
    sys.exit(status - (1 << 32) if status >= 1 << 31 else status)


def _exec(name: str) -> None:
    exe = RELEASE / "bin" / (name + ".exe" if WINDOWS else name)
    if not WINDOWS and exe.is_file() and not os.access(exe, os.X_OK):
        try:
            _restore_modes()
        except OSError:
            pass
    # Windows has no executable bit: what is there runs.
    if not (exe.is_file() if WINDOWS else os.access(exe, os.X_OK)):
        sys.exit(f"{name}: {exe} is missing or not executable; `pip install --force-reinstall superfluid-cli` puts it back")
    if WINDOWS:
        _wait_for(name, exe)
    os.execv(exe, [str(exe), *sys.argv[1:]])


def main() -> None:
    _exec("superfluid")


def workerd() -> None:
    _exec("superfluid-workerd")


def noded() -> None:
    _exec("superfluid-noded")
