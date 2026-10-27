"""Build the PyPI wheel of one superfluid release tarball.

    python3 packaging/python/build_wheel.py <superfluid-<target>.tar.gz> <out-dir> --version <v>

Writes superfluid_cli-<v>-py3-none-<platform>.whl to <out-dir>: the release tree
under superfluid_cli/release/ (packaging/release.py stages it, adding the
installed-by file), and the commands `superfluid`, `superfluid-workerd` and
`superfluid-noded`, launchers that exec the binaries in it
(superfluid_cli/__init__.py). The platform tag is read from the binaries, so
pip refuses the wheel on a machine they cannot run on.

The wheel is written here, not by a build backend: its contents are fixed
files, its binaries must keep their mode (a zip entry's external attributes,
which pip and uv apply), and building it needs nothing from the network.
`superfluid` is taken on PyPI, so the distribution is superfluid-cli.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import release  # noqa: E402

DIST = "superfluid-cli"
PACKAGE = "superfluid_cli"
INSTALLED_BY = f"pip: pip install --upgrade {DIST}"
SUMMARY = "One serving daemon for local LLM inference, across runtimes and machines"
URLS = {
    "Homepage": "https://superfluid.sh",
    "Documentation": "https://superfluid.sh",
    "Source": "https://github.com/basecompute/superfluid",
    "Changelog": "https://github.com/basecompute/superfluid/blob/main/CHANGELOG.md",
    "Issues": "https://github.com/basecompute/superfluid/issues",
}
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
ENTRY_POINTS = {"superfluid": "main", "superfluid-noded": "noded", "superfluid-workerd": "workerd"}


def pep440(version: str) -> str:
    """A release version as PyPI spells it: 0.2.0-rc.1 is 0.2.0rc1."""
    m = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.?(\d+))?", version)
    if not m:
        raise release.ReleaseError(f"{version} is not a version this can publish (X.Y.Z or X.Y.Z-alpha|beta|rc.N)")
    pre = {"alpha": "a", "beta": "b", "rc": "rc"}.get(m.group(2) or "", "")
    return m.group(1) + (f"{pre}{m.group(3)}" if pre else "")


def metadata(version: str) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {DIST}",
        f"Version: {version}",
        f"Summary: {SUMMARY}",
        "Author: Base Compute",
        "License-Expression: Apache-2.0",
        "License-File: LICENSE",
        "License-File: NOTICE",
        "Keywords: llm,inference,serving,llama-cpp,mlx",
        *[f"Classifier: {c}" for c in CLASSIFIERS],
        *[f"Project-URL: {k}, {v}" for k, v in URLS.items()],
        "Requires-Python: >=3.9",
        "Description-Content-Type: text/markdown",
    ]
    return ("\n".join(lines) + "\n\n" + (HERE / "README.md").read_text()).encode()


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def zip_time() -> tuple[int, int, int, int, int, int]:
    """SOURCE_DATE_EPOCH when set, else the zip epoch, so a rebuild of the same release is the same file."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if not epoch:
        return (1980, 1, 1, 0, 0, 0)
    t = time.gmtime(max(int(epoch), 315532800))
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def write_wheel(path: Path, files: list[tuple[str, bytes, int]]) -> None:
    when = zip_time()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data, mode in files:
            info = zipfile.ZipInfo(name, date_time=when)
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)


def build(tarball: Path, out: Path, version: str) -> Path:
    """Write the wheel for one release tarball; return its path."""
    py_version = pep440(version)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "release"
        r = release.stage(tarball, root, version=version, installed_by=INSTALLED_BY)
        tag = f"py3-none-{release.wheel_platform(r.root)}"
        launcher = (HERE / PACKAGE / "__init__.py").read_text()
        if '__version__ = "0+unknown"' not in launcher:
            raise release.ReleaseError("superfluid_cli/__init__.py lost its __version__ placeholder")
        files = [
            (f"{PACKAGE}/__init__.py", launcher.replace('__version__ = "0+unknown"', f'__version__ = "{py_version}"').encode(), 0o644),
            (f"{PACKAGE}/__main__.py", (HERE / PACKAGE / "__main__.py").read_bytes(), 0o644),
        ]
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append((f"{PACKAGE}/release/{p.relative_to(root).as_posix()}", p.read_bytes(), p.stat().st_mode & 0o777))
        info = f"{PACKAGE}-{py_version}.dist-info"
        entry_points = "[console_scripts]\n" + "".join(f"{k} = {PACKAGE}:{v}\n" for k, v in ENTRY_POINTS.items())
        wheel = f"Wheel-Version: 1.0\nGenerator: superfluid packaging/python/build_wheel.py\nRoot-Is-Purelib: false\nTag: {tag}\n"
        files += [
            (f"{info}/METADATA", metadata(py_version), 0o644),
            (f"{info}/WHEEL", wheel.encode(), 0o644),
            (f"{info}/entry_points.txt", entry_points.encode(), 0o644),
            (f"{info}/licenses/LICENSE", (root / "LICENSE").read_bytes(), 0o644),
            (f"{info}/licenses/NOTICE", (root / "NOTICE").read_bytes(), 0o644),
        ]
        record = "".join(record_line(name, data) + "\n" for name, data, _ in files) + f"{info}/RECORD,,\n"
        files.append((f"{info}/RECORD", record.encode(), 0o644))
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{PACKAGE}-{py_version}-{tag}.whl"
        write_wheel(path, files)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tarball", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--version", required=True, help="the release version, without the v")
    a = p.parse_args()
    try:
        print(build(a.tarball, a.out, a.version))
    except release.ReleaseError as e:
        sys.exit(f"build_wheel.py: {e}")


if __name__ == "__main__":
    main()
