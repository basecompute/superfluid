"""Tests for packaging/: the release reader, the wheel, the npm packages, the
Homebrew formula and the two launchers, on small made-up release tarballs.

    python3 -m unittest discover -s tests/scripts -v

Nothing is downloaded; the npm cases need npm and node and are skipped
without them. The real releases are installed by packaging/try-install.sh in
.github/workflows/publish.yml.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging"
for d in (PACKAGING, PACKAGING / "python", PACKAGING / "npm", PACKAGING / "homebrew"):
    sys.path.insert(0, str(d))

import build_packages  # noqa: E402
import build_wheel  # noqa: E402
import formula  # noqa: E402
import own_workers  # noqa: E402
import release  # noqa: E402

TARGET_FORMAT = {
    "aarch64-apple-darwin": ("macho", 0x0100000C),
    "x86_64-unknown-linux-gnu": ("elf", 62),
    "aarch64-unknown-linux-gnu": ("elf", 183),
    "x86_64-pc-windows-msvc": ("pe", 0x8664),
}
WINDOWS = "x86_64-pc-windows-msvc"


def macho(minos=(15, 0), cpu=0x0100000C) -> bytes:
    """A 64-bit Mach-O header with one LC_BUILD_VERSION load command."""
    build_version = struct.pack("<IIIIII", 0x32, 24, 1, (minos[0] << 16) | (minos[1] << 8), 15 << 16, 0)
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cpu, 0, 2, 1, len(build_version), 0, 0)
    return header + build_version + b"\0" * 64


def elf(machine=62, glibc=(17, 28)) -> bytes:
    """A 64-bit little-endian ELF header and the glibc symbol versions it needs."""
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    return ident + struct.pack("<HH", 2, machine) + b"\0" * 44 + b"".join(b"GLIBC_2.%d\x00" % v for v in glibc)


def pe(machine=0x8664, at=0xF8) -> bytes:
    """A DOS header that says where the PE header is, and a PE header for `machine` there, as a linker lays them out."""
    dos = b"MZ" + b"\0" * 58 + struct.pack("<I", at)
    return dos + b"\0" * (at - len(dos)) + b"PE\0\0" + struct.pack("<H", machine) + b"\0" * 64


def binary_for(target: str, **kw) -> bytes:
    fmt, cpu = TARGET_FORMAT[target]
    if fmt == "macho":
        return macho(cpu=cpu, **{k: v for k, v in kw.items() if k == "minos"})
    if fmt == "pe":
        return pe(machine=cpu)
    return elf(machine=cpu, **{k: v for k, v in kw.items() if k == "glibc"})


def make_release(assets: Path, target: str, version: str = "1.2.3", *, files: dict | None = None,
                 extra: list[tarfile.TarInfo] | None = None, sha: str | None = None, **kw) -> Path:
    """Write superfluid-<target>.tar.gz and its .sha256 into assets, laid out as release.yml lays one out."""
    top = f"superfluid-{target}"
    lib = "libsuperfluid_tokenizer_llamacpp." + ("dylib" if target.endswith("darwin") else "so")
    exe = ""
    windows = {}
    if target == WINDOWS:
        # The launcher the install script copies, and the Visual C++ runtime llama.cpp's builds need.
        lib, exe = "superfluid_tokenizer_llamacpp.dll", ".exe"
        windows = {
            "libexec/superfluid/superfluid-shim.exe": binary_for(target),
            "libexec/superfluid/vcruntime/msvcp140.dll": binary_for(target),
            "libexec/superfluid/vcruntime/README.txt": b"Microsoft's\n",
        }
    tree = {
        f"bin/superfluid{exe}": binary_for(target, **kw),
        f"bin/superfluid-workerd{exe}": binary_for(target, **kw),
        f"bin/superfluid-noded{exe}": binary_for(target, **kw),
        f"libexec/superfluid/superfluid-worker-llamacpp{exe}": binary_for(target, **kw),
        f"libexec/superfluid/{lib}": binary_for(target, **kw),
        **windows,
        "compat.json": json.dumps({"superfluid": version, "runtimes": {}}).encode(),
        "LICENSE": b"Apache License 2.0\n",
        "NOTICE": b"superfluid\n",
        "README.md": b"# superfluid\n",
    }
    tree.update(files or {})
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / f"{top}.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for d in (top, f"{top}/bin", f"{top}/libexec", f"{top}/libexec/superfluid"):
            info = tarfile.TarInfo(d)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            tar.addfile(info)
        for name, data in tree.items():
            if data is None:
                continue
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size, info.mode = len(data), 0o644  # modes are set by header, so none are given here
            tar.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tar.addfile(info)
    digest = sha or hashlib.sha256(path.read_bytes()).hexdigest()
    (assets / f"{top}.tar.gz.sha256").write_text(f"{digest}  {top}.tar.gz\n")
    return path


def make_all(assets: Path, version: str = "1.2.3", windows: bool = True) -> None:
    """Every platform's tarball; without `windows`, a release whose Windows build failed."""
    for target in release.TARGETS:
        if windows or target != WINDOWS:
            make_release(assets, target, version)


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="superfluid-packaging-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class ReleaseTest(Scratch):
    def test_a_release_is_staged_as_published_plus_its_installed_by_file_and_modes(self):
        tarball = make_release(self.tmp / "assets", "x86_64-unknown-linux-gnu")
        r = release.stage(tarball, self.tmp / "out", version="1.2.3", installed_by="pip: pip install --upgrade superfluid-cli")
        self.assertEqual((r.target, r.version), ("x86_64-unknown-linux-gnu", "1.2.3"))
        self.assertEqual((r.root / "installed-by").read_text(), "pip: pip install --upgrade superfluid-cli\n")
        mode = lambda p: stat.S_IMODE((r.root / p).stat().st_mode)  # noqa: E731
        for p in ("bin/superfluid", "bin/superfluid-noded", "libexec/superfluid/superfluid-worker-llamacpp", "libexec/superfluid/libsuperfluid_tokenizer_llamacpp.so"):
            self.assertEqual(mode(p), 0o755, p)
        for p in ("compat.json", "LICENSE", "installed-by"):
            self.assertEqual(mode(p), 0o644, p)
        self.assertEqual(release.wheel_platform(r.root), "manylinux_2_28_x86_64")

    def test_the_platform_tag_comes_from_the_binaries(self):
        cases = [
            ("aarch64-apple-darwin", {"minos": (15, 0)}, "macosx_15_0_arm64"),
            ("aarch64-apple-darwin", {"minos": (15, 4)}, "macosx_16_0_arm64"),
            ("aarch64-apple-darwin", {"minos": (11, 0)}, "macosx_11_0_arm64"),
            ("aarch64-unknown-linux-gnu", {"glibc": (17, 28)}, "manylinux_2_28_aarch64"),
            ("x86_64-unknown-linux-gnu", {"glibc": (2, 34, 17)}, "manylinux_2_34_x86_64"),
        ]
        for i, (target, kw, tag) in enumerate(cases):
            with self.subTest(tag=tag):
                r = release.stage(make_release(self.tmp / f"a{i}", target, **kw), self.tmp / f"o{i}")
                self.assertEqual(release.wheel_platform(r.root), tag)

    def test_a_tarball_that_does_not_check_out_or_holds_more_than_files_is_refused(self):
        link = tarfile.TarInfo("superfluid-aarch64-unknown-linux-gnu/libexec/superfluid/libfoo.so")
        link.type, link.linkname = tarfile.SYMTYPE, "libfoo.so.1"
        escape = tarfile.TarInfo("superfluid-aarch64-unknown-linux-gnu/../evil")
        outside = tarfile.TarInfo("elsewhere/file")
        apple = tarfile.TarInfo("._superfluid-aarch64-unknown-linux-gnu")
        cases = {
            "SHA-256": dict(sha="0" * 64),
            "symlink": dict(extra=[link]),
            "outside": dict(extra=[escape]),
            "elsewhere/file": dict(extra=[outside]),
            "._superfluid": dict(extra=[apple]),
            "no bin/superfluid-noded": dict(files={"bin/superfluid-noded": None}),
            "not elf aarch64": dict(files={"libexec/superfluid/superfluid-worker-llamacpp": elf(machine=62)}),
            "not an executable": dict(files={"bin/superfluid": b"#!/bin/sh\n"}),
            "no runtime workers": dict(files={"libexec/superfluid/superfluid-worker-llamacpp": None}),
            "already has an installed-by": dict(files={"installed-by": b"pip: x\n"}),
            "no NOTICE": dict(files={"NOTICE": None}),
        }
        for i, (why, kw) in enumerate(cases.items()):
            with self.subTest(why=why):
                tarball = make_release(self.tmp / f"a{i}", "aarch64-unknown-linux-gnu", **kw)
                with self.assertRaises(release.ReleaseError) as e:
                    release.stage(tarball, self.tmp / f"o{i}")
                self.assertIn(why, str(e.exception))
        tarball = make_release(self.tmp / "v", "aarch64-unknown-linux-gnu", "1.2.3")
        with self.assertRaisesRegex(release.ReleaseError, "superfluid 1.2.3, not 1.2.4"):
            release.stage(tarball, self.tmp / "ov", version="1.2.4")
        (self.tmp / "v" / f"{tarball.name}.sha256").unlink()
        with self.assertRaisesRegex(release.ReleaseError, "no .*sha256"):
            release.stage(tarball, self.tmp / "ov2")
        with self.assertRaisesRegex(release.ReleaseError, "not a release tarball"):
            release.target_of(Path("superfluid-x86_64-apple-darwin.tar.gz"))

    def test_a_windows_release_is_read_by_its_pe_headers_and_exe_names(self):
        r = release.stage(make_release(self.tmp / "assets", WINDOWS), self.tmp / "out", version="1.2.3", installed_by="npm: x")
        self.assertEqual(r.target, WINDOWS)
        found = {p.relative_to(r.root).as_posix() for p in release.binaries(r.root)}
        self.assertEqual(found, {
            "bin/superfluid.exe", "bin/superfluid-workerd.exe", "bin/superfluid-noded.exe",
            "libexec/superfluid/superfluid-worker-llamacpp.exe", "libexec/superfluid/superfluid_tokenizer_llamacpp.dll",
            "libexec/superfluid/superfluid-shim.exe", "libexec/superfluid/vcruntime/msvcp140.dll",
        })
        self.assertEqual(release.wheel_platform(r.root), "win_amd64")
        self.assertEqual([release.command_file(WINDOWS, c) for c in release.COMMANDS], ["superfluid.exe", "superfluid-workerd.exe", "superfluid-noded.exe"])
        self.assertEqual(release.command_file("aarch64-apple-darwin", "superfluid"), "superfluid")
        # A linker may put the PE header anywhere in the first pages; a DOS program with none is not one of ours.
        self.assertEqual(release.binary_kind(pe(at=0x40)), ("pe", "x86_64"))
        self.assertEqual(release.binary_kind(pe(at=0x200, machine=0xAA64)), ("pe", "arm64"))
        self.assertIsNone(release.binary_kind(b"MZ" + b"\0" * 200))
        self.assertIsNone(release.binary_kind(b"MZ is how this text file starts, and it goes on for more than sixty-four bytes"))
        cases = {
            "no bin/superfluid.exe": {"bin/superfluid.exe": None},
            "not pe x86_64": {"libexec/superfluid/vcruntime/msvcp140.dll": pe(machine=0xAA64)},
            "is elf x86_64": {"libexec/superfluid/superfluid-worker-llamacpp.exe": elf()},
            "bin/superfluid-noded.exe is not an executable": {"bin/superfluid-noded.exe": b"MZ" + b"\0" * 200},
            "no runtime workers": {"libexec/superfluid/superfluid-worker-llamacpp.exe": None},
        }
        for i, (why, files) in enumerate(cases.items()):
            with self.subTest(why=why):
                with self.assertRaises(release.ReleaseError) as e:
                    release.stage(make_release(self.tmp / f"a{i}", WINDOWS, files=files), self.tmp / f"o{i}")
                self.assertIn(why, str(e.exception))

    def test_a_release_without_its_windows_tarball_is_packaged_for_the_rest(self):
        make_all(self.tmp / "with")
        self.assertEqual(release.targets_in(self.tmp / "with"), list(release.TARGETS))
        make_all(self.tmp / "without", windows=False)
        self.assertEqual(release.targets_in(self.tmp / "without"), [t for t in release.TARGETS if t != WINDOWS])
        # Only what release.yml lets a tag go out without is left out: the others are asked for and missed.
        (self.tmp / "without" / "superfluid-aarch64-apple-darwin.tar.gz").unlink()
        self.assertIn("aarch64-apple-darwin", release.targets_in(self.tmp / "without"))

    def test_installed_by_must_be_one_manager_and_one_command(self):
        for i, bad in enumerate(["pip", "pip:", ": cmd", "pip: a\nb"]):
            with self.subTest(bad=bad), self.assertRaises(release.ReleaseError):
                release.stage(make_release(self.tmp / f"a{i}", "aarch64-apple-darwin"), self.tmp / f"o{i}", installed_by=bad)


class ContractTest(unittest.TestCase):
    """What packaging/ assumes of the daemon and of release.yml, read from them."""

    def test_the_commands_and_the_installed_by_file_are_the_daemons(self):
        src = (ROOT / "crates/superfluid-daemon/src/selfupdate.rs").read_text()
        linked = re.search(r"const LINKED: &\[&str\] = &\[(.*?)\];", src).group(1)
        self.assertEqual(tuple(re.findall(r'"([^"]+)"', linked)), release.COMMANDS)
        self.assertIn(f'pub const INSTALLED_BY: &str = "{release.INSTALLED_BY}";', src)
        for line in (build_wheel.INSTALLED_BY, build_packages.INSTALLED_BY, formula.INSTALLED_BY):
            self.assertRegex(line, r"^[^:\n]+: \S.*$", "parsed by package_manager() in selfupdate.rs")

    def test_every_target_release_yml_builds_is_packaged(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertEqual(set(re.findall(r"^\s+target: (\S+)$", workflow, re.M)), set(release.TARGETS))
        self.assertEqual(set(build_packages.TARGETS), set(release.TARGETS))
        self.assertLessEqual(set(release.OPTIONAL), set(release.TARGETS))
        # The formula has a url for each platform Homebrew runs on, and none may be missing from a release.
        self.assertEqual(set(formula.TARGETS), set(release.TARGETS) - set(release.OPTIONAL))
        # A target release.yml lets a tag go out without is the one it says so of.
        for target in release.OPTIONAL:
            self.assertRegex(workflow, rf"continue-on-error: \$\{{\{{ matrix\.target == '{re.escape(target)}'")

    def test_what_release_yml_tries_on_windows_is_what_a_linux_build_publishes(self):
        # The Windows packages are tried where their tarball was built, by these builders run on Windows. Python
        # there reads text in the machine's code page and writes CRLF, unless each read and write says otherwise.
        for name in ("release.py", "build.py", "python/build_wheel.py", "npm/build_packages.py"):
            source = (ROOT / "packaging" / name).read_text(encoding="utf-8")
            self.assertNotIn(".read_text()", source, f"{name} reads text in the machine's own encoding")
            self.assertNotIn(".write_text(", source, f"{name} writes text with the machine's own line ends")
        with tempfile.TemporaryDirectory() as tmp:
            release.put_text(Path(tmp) / "f", "a \u00b7 b\n")
            self.assertEqual((Path(tmp) / "f").read_bytes(), b"a \xc2\xb7 b\n")
        # The runner's jq ends its line with CRLF, and a version with a carriage return is none: the step that
        # reads it drops the CR, and the builder refuses one that kept it.
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        step = workflow.split("- name: Package the release for pip and npm, and install each")[1].split("\n      - ")[0]
        self.assertIn("shell: bash", step)
        reads = [line for line in step.splitlines() if line.strip().startswith('v="$(')]
        self.assertEqual(len(reads), 1, step)
        self.assertIn("| tr -d '\\r')\"", reads[0])
        with self.assertRaisesRegex(release.ReleaseError, "not a version"):
            build_wheel.pep440("0.2.0\r")

    def test_publish_yml_asks_the_registries_for_every_package_and_wheel(self):
        workflow = (ROOT / ".github/workflows/publish.yml").read_text()
        for t in build_packages.TARGETS.values():
            name = t["name"].split("/")[1]
            self.assertGreaterEqual(workflow.count(name), 2, f"{name}: what npm has, and the upload")
        for tag in ("macosx_*_arm64", "manylinux_*_x86_64", "manylinux_*_aarch64", "win_amd64"):
            self.assertIn(f"py3-none-{tag}.whl", workflow)
        for target in release.OPTIONAL:
            self.assertIn(f"superfluid-{target}".replace(".", "\\."), workflow, "the plan looks for an optional target's tarball")

    def test_the_npm_launcher_names_the_packages_the_builder_makes(self):
        js = (PACKAGING / "npm/launch.js").read_text()
        named = dict(re.findall(r'"([a-z0-9-]+)": "(@basecompute/superfluid-[a-z0-9-]+)"', js))
        self.assertEqual(set(named.values()), {t["name"] for t in build_packages.TARGETS.values()})
        for key, name in named.items():
            self.assertTrue(name.endswith(key), f"{key} -> {name}")


class WheelTest(Scratch):
    def test_the_wheel_carries_the_release_its_commands_and_a_valid_record(self):
        tarball = make_release(self.tmp / "assets", "aarch64-apple-darwin", "1.2.3")
        whl = build_wheel.build(tarball, self.tmp / "out", "1.2.3")
        self.assertEqual(whl.name, "superfluid_cli-1.2.3-py3-none-macosx_15_0_arm64.whl")
        with zipfile.ZipFile(whl) as z:
            names = z.namelist()
            info = "superfluid_cli-1.2.3.dist-info"
            self.assertEqual(names[-1], f"{info}/RECORD")
            record = z.read(f"{info}/RECORD").decode().splitlines()
            self.assertEqual(len(record), len(names))
            for line in record:
                name, digest, size = line.rsplit(",", 2)
                if name.endswith("RECORD"):
                    self.assertEqual((digest, size), ("", ""))
                    continue
                data = z.read(name)
                want = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                self.assertEqual((digest, int(size)), (want, len(data)), name)
            modes = {i.filename: (i.external_attr >> 16) & 0o777 for i in z.infolist()}
            self.assertEqual(modes["superfluid_cli/release/bin/superfluid"], 0o755)
            self.assertEqual(modes["superfluid_cli/release/libexec/superfluid/superfluid-worker-llamacpp"], 0o755)
            self.assertEqual(modes["superfluid_cli/release/compat.json"], 0o644)
            self.assertEqual(z.read("superfluid_cli/release/installed-by").decode(), "pip: pip install --upgrade superfluid-cli\n")
            self.assertIn('__version__ = "1.2.3"', z.read("superfluid_cli/__init__.py").decode())
            entry_points = z.read(f"{info}/entry_points.txt").decode()
            for command in release.COMMANDS:
                self.assertRegex(entry_points, rf"(?m)^{command} = superfluid_cli:\w+$")
            meta = z.read(f"{info}/METADATA").decode()
            for field in ("Metadata-Version: 2.4", "Name: superfluid-cli", "Version: 1.2.3", "License-Expression: Apache-2.0", "License-File: NOTICE"):
                self.assertIn(field + "\n", meta)
            self.assertIn(f"{info}/licenses/NOTICE", names)
            self.assertIn("Tag: py3-none-macosx_15_0_arm64\n", z.read(f"{info}/WHEEL").decode())
            self.assertIn("Root-Is-Purelib: false\n", z.read(f"{info}/WHEEL").decode())

    def test_a_windows_wheel_is_tagged_for_windows_and_carries_the_exe_files(self):
        whl = build_wheel.build(make_release(self.tmp / "assets", WINDOWS), self.tmp / "out", "1.2.3")
        self.assertEqual(whl.name, "superfluid_cli-1.2.3-py3-none-win_amd64.whl")
        with zipfile.ZipFile(whl) as z:
            names = set(z.namelist())
            for name in ("bin/superfluid.exe", "bin/superfluid-workerd.exe", "bin/superfluid-noded.exe", "libexec/superfluid/superfluid-worker-llamacpp.exe",
                         "libexec/superfluid/superfluid_tokenizer_llamacpp.dll", "libexec/superfluid/vcruntime/msvcp140.dll", "installed-by"):
                self.assertIn(f"superfluid_cli/release/{name}", names)
            info = "superfluid_cli-1.2.3.dist-info"
            self.assertIn("Tag: py3-none-win_amd64\n", z.read(f"{info}/WHEEL").decode())
            # pip makes Scripts\superfluid.exe and its siblings from these, as it makes bin/superfluid elsewhere.
            for command in release.COMMANDS:
                self.assertRegex(z.read(f"{info}/entry_points.txt").decode(), rf"(?m)^{command} = superfluid_cli:\w+$")
            self.assertIn("Classifier: Operating System :: Microsoft :: Windows\n", z.read(f"{info}/METADATA").decode())

    def test_the_same_release_makes_the_same_wheel(self):
        tarball = make_release(self.tmp / "assets", "x86_64-unknown-linux-gnu")
        a = build_wheel.build(tarball, self.tmp / "a", "1.2.3").read_bytes()
        b = build_wheel.build(tarball, self.tmp / "b", "1.2.3").read_bytes()
        self.assertEqual(a, b)

    def test_versions_are_spelled_as_pypi_spells_them(self):
        self.assertEqual(build_wheel.pep440("0.1.4"), "0.1.4")
        self.assertEqual(build_wheel.pep440("0.2.0-rc.1"), "0.2.0rc1")
        self.assertEqual(build_wheel.pep440("0.2.0-beta.3"), "0.2.0b3")
        with self.assertRaises(release.ReleaseError):
            build_wheel.pep440("0.2.0-nightly")


class LauncherTest(Scratch):
    """The launchers run a stand-in release: a shell script where superfluid would be."""

    STAND_IN = "#!/bin/sh\necho \"ran $(basename \"$0\") $*\"\nexit 7\n"

    def stand_in_release(self, release_dir: Path, executable=True, exe="") -> None:
        (release_dir / "bin").mkdir(parents=True)
        (release_dir / "libexec" / "superfluid").mkdir(parents=True)
        for command in release.COMMANDS:
            script = release_dir / "bin" / (command + exe)
            script.write_text(self.STAND_IN)
            script.chmod(0o755 if executable else 0o644)

    # A stand-in that records each signal it gets and exits as a shell does for
    # it. It says "ready" once the sleep it waits on has a pid, so a signal that
    # follows finds $pid set, and the sleep ends with it: left behind, it would
    # outlive the test, and its output goes nowhere so it can never hold the
    # launcher's pipes.
    def waiting_stand_in(self, script: Path, log: Path) -> None:
        script.write_text(
            "#!/bin/sh\n"
            f'trap \'echo INT >> "{log}"; kill $pid; exit 130\' INT\n'
            f'trap \'echo TERM >> "{log}"; kill $pid; exit 143\' TERM\n'
            "sleep 30 > /dev/null 2>&1 & pid=$!\n"
            "echo ready\n"
            "wait $pid\n"
        )

    def launched(self, argv: list[str], **kw) -> subprocess.Popen:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
        self.addCleanup(p.kill)
        self.assertEqual(p.stdout.readline(), "ready\n")
        return p

    def ended(self, p: subprocess.Popen, log: Path) -> tuple[int, list[str], str]:
        _, err = p.communicate(timeout=10)
        got = log.read_text().split() if log.is_file() else []
        log.unlink(missing_ok=True)
        return p.returncode, got, err

    def test_the_python_launcher_execs_the_release_with_the_arguments_and_status(self):
        pkg = self.tmp / "site" / "superfluid_cli"
        shutil.copytree(PACKAGING / "python" / "superfluid_cli", pkg)
        self.stand_in_release(pkg / "release")
        env = {**os.environ, "PYTHONPATH": str(self.tmp / "site")}
        for entry, command in (("main", "superfluid"), ("workerd", "superfluid-workerd"), ("noded", "superfluid-noded")):
            p = subprocess.run([sys.executable, "-c", f"import superfluid_cli; superfluid_cli.{entry}()", "a b", "--c"],
                               env=env, capture_output=True, text=True)
            self.assertEqual((p.returncode, p.stdout), (7, f"ran {command} a b --c\n"))
        p = subprocess.run([sys.executable, "-m", "superfluid_cli", "--version"], env=env, capture_output=True, text=True)
        self.assertEqual(p.stdout, "ran superfluid --version\n")

    def test_the_python_launcher_on_windows_waits_for_the_program_and_leaves_ctrl_c_to_it(self):
        # Windows has no exec, so there the launcher runs the program and waits.
        # That path is taken here by saying so (WINDOWS), with a shell script
        # under the .exe name; what only Windows does is tried in publish.yml.
        pkg = self.tmp / "site" / "superfluid_cli"
        shutil.copytree(PACKAGING / "python" / "superfluid_cli", pkg)
        self.stand_in_release(pkg / "release", exe=".exe")
        env = {**os.environ, "PYTHONPATH": str(self.tmp / "site")}
        run = "import superfluid_cli; superfluid_cli.WINDOWS = True; superfluid_cli.{}()"
        for entry, command in (("main", "superfluid"), ("workerd", "superfluid-workerd"), ("noded", "superfluid-noded")):
            p = subprocess.run([sys.executable, "-c", run.format(entry), "a b", "--c"], env=env, capture_output=True, text=True)
            self.assertEqual((p.returncode, p.stdout, p.stderr), (7, f"ran {command}.exe a b --c\n", ""))
        # Ctrl+C reaches the launcher and the program; the program answers it, and
        # the launcher waits and exits with its status, with no traceback.
        log = self.tmp / "signals"
        self.waiting_stand_in(pkg / "release" / "bin" / "superfluid.exe", log)
        p = self.launched([sys.executable, "-c", run.format("main")], env=env, start_new_session=True)
        os.killpg(p.pid, signal.SIGINT)
        code, got, err = self.ended(p, log)
        self.assertEqual((code, got, err), (130, ["INT"], ""))
        # A release with no program in it is said to be broken, not run.
        (pkg / "release" / "bin" / "superfluid.exe").unlink()
        p = subprocess.run([sys.executable, "-c", run.format("main")], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("pip install --force-reinstall superfluid-cli", p.stderr)

    def test_the_python_launcher_says_how_to_repair_a_release_it_cannot_run(self):
        pkg = self.tmp / "site" / "superfluid_cli"
        shutil.copytree(PACKAGING / "python" / "superfluid_cli", pkg)
        self.stand_in_release(pkg / "release", executable=False)
        p = subprocess.run([sys.executable, "-c", "import superfluid_cli; superfluid_cli.main()"],
                           env={**os.environ, "PYTHONPATH": str(self.tmp / "site")}, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("pip install --force-reinstall superfluid-cli", p.stderr)

    def npm_fixture(self) -> tuple[Path, Path, str]:
        """node_modules as npm lays it out for this machine: the launcher's bin/superfluid-noded.js, the platform package, its name."""
        key = {("darwin", "arm64"), ("linux", "x86_64"), ("linux", "aarch64")}
        machine = (sys.platform if sys.platform != "linux" else "linux", os.uname().machine)
        if machine not in key:
            self.skipTest(f"no release for {machine}")
        npm_platform = {("darwin", "arm64"): "darwin-arm64", ("linux", "x86_64"): "linux-x64", ("linux", "aarch64"): "linux-arm64"}[machine]
        modules = self.tmp / "node_modules" / "@basecompute"
        platform_pkg = modules / f"superfluid-{npm_platform}"
        self.stand_in_release(platform_pkg / "release")
        (platform_pkg / "package.json").write_text(json.dumps({"name": f"@basecompute/superfluid-{npm_platform}", "version": "1.2.3"}))
        (modules / "superfluid" / "lib").mkdir(parents=True)
        shutil.copyfile(PACKAGING / "npm" / "launch.js", modules / "superfluid" / "lib" / "launch.js")
        (modules / "superfluid" / "bin").mkdir()
        bin_js = modules / "superfluid" / "bin" / "superfluid-noded.js"
        bin_js.write_text('require("../lib/launch.js")("superfluid-noded");\n')
        return bin_js, platform_pkg, f"@basecompute/superfluid-{npm_platform}"

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_the_npm_launcher_runs_this_platforms_package(self):
        bin_js, platform_pkg, name = self.npm_fixture()
        p = subprocess.run(["node", str(bin_js), "a b", "--c"], capture_output=True, text=True)
        self.assertEqual((p.returncode, p.stdout), (7, "ran superfluid-noded a b --c\n"), p.stderr)
        shutil.rmtree(platform_pkg)
        p = subprocess.run(["node", str(bin_js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"{name} is not installed", p.stderr)

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_the_npm_launcher_passes_on_signals_from_a_terminal_and_from_kill(self):
        # The stand-in records each signal it gets and exits as a shell does
        # for it. A terminal's Ctrl-C reaches the whole foreground group; a
        # supervisor's kill(2), or Popen.send_signal, reaches the launcher alone.
        bin_js, platform_pkg, _ = self.npm_fixture()
        log = self.tmp / "signals"
        self.waiting_stand_in(platform_pkg / "release" / "bin" / "superfluid-noded", log)
        launch = lambda **kw: self.launched(["node", str(bin_js)], **kw)  # noqa: E731
        ended = lambda p: self.ended(p, log)  # noqa: E731

        # kill(2) to the launcher's pid alone, as a supervisor or a script sends it.
        p = launch()
        p.send_signal(signal.SIGINT)
        code, got, err = ended(p)
        self.assertEqual((code, got), (130, ["INT"]), err)
        p = launch()
        p.send_signal(signal.SIGTERM)
        code, got, err = ended(p)
        self.assertEqual((code, got), (143, ["TERM"]), err)
        # The whole process group, as a terminal delivers Ctrl-C: the child
        # gets its own copy and the launcher's, and ends once.
        p = launch(start_new_session=True)
        os.killpg(p.pid, signal.SIGINT)
        code, got, err = ended(p)
        self.assertEqual((code, got[:1]), (130, ["INT"]), err)


    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_the_npm_launcher_on_windows_runs_the_exe_and_passes_nothing_on(self):
        # The launcher is told it is on Windows, with a shell script under the
        # .exe name; what only Windows does is tried in publish.yml.
        modules = self.tmp / "node_modules" / "@basecompute"
        platform_pkg = modules / "superfluid-win32-x64"
        self.stand_in_release(platform_pkg / "release", exe=".exe")
        (platform_pkg / "package.json").write_text(json.dumps({"name": "@basecompute/superfluid-win32-x64", "version": "1.2.3"}))
        (modules / "superfluid" / "lib").mkdir(parents=True)
        shutil.copyfile(PACKAGING / "npm" / "launch.js", modules / "superfluid" / "lib" / "launch.js")
        (modules / "superfluid" / "bin").mkdir()
        bin_js = modules / "superfluid" / "bin" / "superfluid-noded.js"
        bin_js.write_text('require("../lib/launch.js")("superfluid-noded", "win32", "x64");\n')
        p = subprocess.run(["node", str(bin_js), "a b", "--c"], capture_output=True, text=True)
        self.assertEqual((p.returncode, p.stdout), (7, "ran superfluid-noded.exe a b --c\n"), p.stderr)
        # Ctrl+C reaches the launcher and the program alike: the program answers
        # it, and the launcher, which there would end the program outright by
        # passing it on, only waits and exits with its status.
        log = self.tmp / "signals"
        self.waiting_stand_in(platform_pkg / "release" / "bin" / "superfluid-noded.exe", log)
        p = self.launched(["node", str(bin_js)], start_new_session=True)
        os.killpg(p.pid, signal.SIGINT)
        code, got, err = self.ended(p, log)
        self.assertEqual((code, got), (130, ["INT"]), err)
        # One sent to the launcher alone ends nothing; the program goes on until its own arrives.
        p = self.launched(["node", str(bin_js)], start_new_session=True)
        p.send_signal(signal.SIGINT)
        with self.assertRaises(subprocess.TimeoutExpired):
            p.wait(timeout=1)
        self.assertFalse(log.exists())
        os.killpg(p.pid, signal.SIGTERM)
        p.communicate(timeout=10)
        bin_js.write_text('require("../lib/launch.js")("superfluid-noded", "win32", "arm64");\n')
        p = subprocess.run(["node", str(bin_js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("no release for win32-arm64", p.stderr)


def manifest(tgz: Path) -> tuple[dict, dict]:
    """An npm tarball's package.json and its members by name."""
    with tarfile.open(tgz) as tar:
        return json.load(tar.extractfile("package/package.json")), {m.name: m for m in tar}


@unittest.skipUnless(shutil.which("npm"), "needs npm")
class NpmTest(Scratch):
    def test_a_package_for_each_platform_restricted_to_it_and_the_main_one_pointing_at_them(self):
        make_all(self.tmp / "assets")
        built = build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3")
        self.assertEqual([p.name for p in built], [
            "basecompute-superfluid-darwin-arm64-1.2.3.tgz",
            "basecompute-superfluid-linux-x64-1.2.3.tgz",
            "basecompute-superfluid-linux-arm64-1.2.3.tgz",
            "basecompute-superfluid-win32-x64-1.2.3.tgz",
            "basecompute-superfluid-1.2.3.tgz",
        ])

        linux, members = manifest(built[1])
        self.assertEqual((linux["name"], linux["os"], linux["cpu"], linux["libc"]), ("@basecompute/superfluid-linux-x64", ["linux"], ["x64"], ["glibc"]))
        self.assertEqual(members["package/release/bin/superfluid"].mode & 0o777, 0o755)
        self.assertEqual(members["package/release/compat.json"].mode & 0o777, 0o644)
        for name in ("package/release/installed-by", "package/LICENSE", "package/NOTICE", "package/README.md"):
            self.assertIn(name, members)
        mac, _ = manifest(built[0])
        self.assertNotIn("libc", mac)
        win, members = manifest(built[3])
        self.assertEqual((win["name"], win["os"], win["cpu"]), ("@basecompute/superfluid-win32-x64", ["win32"], ["x64"]))
        self.assertNotIn("libc", win)
        for name in ("bin/superfluid.exe", "bin/superfluid-noded.exe", "libexec/superfluid/superfluid-worker-llamacpp.exe",
                     "libexec/superfluid/vcruntime/msvcp140.dll", "installed-by"):
            self.assertIn(f"package/release/{name}", members)
        main, members = manifest(built[4])
        self.assertEqual(main["name"], "@basecompute/superfluid")
        self.assertEqual(main["optionalDependencies"], {t["name"]: "1.2.3" for t in build_packages.TARGETS.values()})
        self.assertEqual((main["os"], main["cpu"]), (["darwin", "linux", "win32"], ["arm64", "x64"]))
        self.assertEqual(main["bin"], {c: f"bin/{c}.js" for c in release.COMMANDS})
        for command in release.COMMANDS:
            self.assertEqual(members[f"package/bin/{command}.js"].mode & 0o777, 0o755)
        self.assertIn("package/lib/launch.js", members)

    def test_a_release_without_its_windows_tarball_makes_packages_that_do_not_name_windows(self):
        make_all(self.tmp / "assets", windows=False)
        built = build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3")
        self.assertEqual(len(built), 4)
        main, _ = manifest(built[-1])
        self.assertEqual(sorted(main["optionalDependencies"]), ["@basecompute/superfluid-darwin-arm64", "@basecompute/superfluid-linux-arm64", "@basecompute/superfluid-linux-x64"])
        # npm refuses it on Windows, where it has nothing to run.
        self.assertEqual(main["os"], ["darwin", "linux"])

    def test_one_platform_alone_is_packaged_when_named(self):
        # release.yml tries the Windows packages where the tarball was built, with no other platform's beside it.
        make_release(self.tmp / "assets", WINDOWS)
        built = build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3", [WINDOWS])
        self.assertEqual([p.name for p in built], ["basecompute-superfluid-win32-x64-1.2.3.tgz", "basecompute-superfluid-1.2.3.tgz"])
        main, members = manifest(built[-1])
        self.assertEqual((main["os"], main["cpu"], main["optionalDependencies"]), (["win32"], ["x64"], {"@basecompute/superfluid-win32-x64": "1.2.3"}))
        self.assertIn("package/LICENSE", members)

    def test_a_release_missing_a_platform_makes_no_packages(self):
        make_release(self.tmp / "assets", "aarch64-apple-darwin")
        with self.assertRaisesRegex(release.ReleaseError, "no superfluid-x86_64-unknown-linux-gnu.tar.gz"):
            build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3")


class OwnWorkersTest(Scratch):
    """try-install.sh's check of `superfluid runtimes --json`, on both kinds of release."""

    def release(self, *files: str) -> Path:
        root = self.tmp / "pkg" / "release"
        for name in files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"")
        return root

    @staticmethod
    def row(rid: str, source: str, path) -> dict:
        return {"id": rid, "state": "not installed", "worker": {"path": str(path), "source": source}}

    def test_a_release_that_ships_its_adapters_runs_them_from_libexec(self):
        root = self.release("bin/superfluid-workerd", "libexec/superfluid/superfluid-worker-llamacpp", "libexec/superfluid/superfluid-worker-mlx")
        rows = [
            self.row("llamacpp", "shipped with superfluid", root / "libexec/superfluid/superfluid-worker-llamacpp"),
            self.row("mlx", "shipped with superfluid", root / "libexec/superfluid/superfluid-worker-mlx"),
            # A runtime installed on the machine is not the package's, wherever it is.
            self.row("toyrt", "installed", self.tmp / "home/runtimes/toyrt/1/bin/superfluid-worker-toyrt"),
            {"id": "basert", "state": "not installed"},
        ]
        self.assertEqual(own_workers.own_workers(rows, root), {"llamacpp": "shipped with superfluid", "mlx": "shipped with superfluid"})

    def test_the_windows_release_runs_llama_cpp_built_into_its_own_workerd(self):
        # Built with `--no-default-features --features llamacpp,tui,shim`: llama.cpp is linked into the daemon, and
        # `runtimes` names the package's superfluid-workerd.exe, with the adapter files in libexec all the same.
        root = self.release("bin/superfluid.exe", "bin/superfluid-workerd.exe", "libexec/superfluid/superfluid-worker-llamacpp.exe", "libexec/superfluid/superfluid-shim.exe")
        rows = [self.row("llamacpp", "built in", root / "bin" / "superfluid-workerd.exe")]
        self.assertEqual(own_workers.own_workers(rows, root), {"llamacpp": "built in"})
        # As try-install.sh runs it, on what the installed superfluid printed.
        p = subprocess.run([sys.executable, str(PACKAGING / "own_workers.py"), str(root)], input=json.dumps(rows), capture_output=True, text=True)
        self.assertEqual((p.returncode, p.stdout, p.stderr), (0, "workers: llamacpp (built in)\n", ""))

    def test_a_worker_from_outside_the_package_or_none_for_llama_cpp_is_refused(self):
        root = self.release("bin/superfluid-workerd", "libexec/superfluid/superfluid-worker-llamacpp")
        other = self.tmp / "other" / "release"
        for name in ("bin/superfluid-workerd", "libexec/superfluid/superfluid-worker-llamacpp"):
            (other / name).parent.mkdir(parents=True, exist_ok=True)
            (other / name).write_bytes(b"")
        shipped = self.row("llamacpp", "shipped with superfluid", root / "libexec/superfluid/superfluid-worker-llamacpp")
        for rows, why in (
            ([self.row("llamacpp", "shipped with superfluid", other / "libexec/superfluid/superfluid-worker-llamacpp")], "not in"),
            ([self.row("llamacpp", "built in", other / "bin/superfluid-workerd")], "not in"),
            # A built-in worker is the daemon's own program, not something in libexec, and the other way about.
            ([self.row("llamacpp", "built in", root / "libexec/superfluid/superfluid-worker-llamacpp")], "not in"),
            ([self.row("llamacpp", "shipped with superfluid", root / "libexec/superfluid/gone")], "is not there"),
            ([self.row("llamacpp", "installed", self.tmp / "home/runtimes/llamacpp/b1/bin/superfluid-worker-llamacpp")], "no llama.cpp worker"),
            ([shipped, self.row("mlx", "shipped with superfluid", other / "libexec/superfluid/superfluid-worker-mlx")], "the mlx worker"),
            ([], "no llama.cpp worker"),
        ):
            with self.subTest(rows=rows), self.assertRaisesRegex(own_workers.NotOwn, why):
                own_workers.own_workers(rows, root)
        # The Windows release finds llama.cpp in the daemon, and still has to carry what it ships in libexec.
        bare = self.tmp / "bare" / "release"
        (bare / "bin").mkdir(parents=True)
        (bare / "bin" / "superfluid-workerd.exe").write_bytes(b"")
        (bare / "libexec" / "superfluid").mkdir(parents=True)
        with self.assertRaisesRegex(own_workers.NotOwn, "holds nothing"):
            own_workers.own_workers([self.row("llamacpp", "built in", bare / "bin" / "superfluid-workerd.exe")], bare)
        p = subprocess.run([sys.executable, str(PACKAGING / "own_workers.py"), str(root)], input="[]", capture_output=True, text=True)
        self.assertEqual((p.returncode, p.stdout), (1, ""))
        self.assertIn("no llama.cpp worker of the package's own", p.stderr)

    def test_the_sources_are_the_daemons_names_and_try_install_runs_this_check(self):
        src = (ROOT / "crates/superfluid-daemon/src/runtimes.rs").read_text(encoding="utf-8")
        for source in own_workers.OWN:
            self.assertIn(f'=> "{source}",', src)
        script = (PACKAGING / "try-install.sh").read_text(encoding="utf-8")
        self.assertIn('runtimes --json | "$python" "$here/own_workers.py" "$where"', script)


class FormulaTest(Scratch):
    def test_the_formula_names_each_tarball_its_checksum_and_the_macos_it_needs(self):
        make_all(self.tmp / "assets")
        rb = formula.build(self.tmp / "assets", self.tmp / "out", "1.2.3").read_text()
        self.assertNotIn("windows", rb, "Homebrew has no Windows")
        for target in formula.TARGETS:
            self.assertIn(f'url "https://github.com/basecompute/superfluid/releases/download/v1.2.3/superfluid-{target}.tar.gz"', rb)
            digest = (self.tmp / "assets" / f"superfluid-{target}.tar.gz.sha256").read_text().split()[0]
            self.assertIn(f'sha256 "{digest}"', rb)
        self.assertIn("depends_on macos: :sequoia", rb)
        self.assertIn('(libexec/"installed-by").write "Homebrew: brew upgrade superfluid\\n"', rb)
        self.assertNotIn("{", rb.replace("#{", ""), "every placeholder was filled")

    def test_a_minimum_between_named_macos_versions_takes_the_next_one(self):
        shas = {t: "0" * 64 for t in formula.TARGETS}
        self.assertIn(":sequoia", formula.render("1.2.3", shas, (15, 0)))
        self.assertIn(":tahoe", formula.render("1.2.3", shas, (15, 4)))
        self.assertIn(":sonoma", formula.render("1.2.3", shas, (13, 5)))
        with self.assertRaises(release.ReleaseError):
            formula.render("1.2.3", shas, (27, 0))

    def test_a_checksum_that_does_not_match_its_tarball_makes_no_formula(self):
        make_all(self.tmp / "assets")
        make_release(self.tmp / "assets", "aarch64-unknown-linux-gnu", sha="f" * 64)
        with self.assertRaisesRegex(release.ReleaseError, "SHA-256"):
            formula.build(self.tmp / "assets", self.tmp / "out", "1.2.3")


class BuildTest(Scratch):
    @unittest.skipUnless(shutil.which("npm"), "needs npm")
    def test_build_writes_every_package_of_a_release(self):
        make_all(self.tmp / "assets", "2.0.0")
        p = subprocess.run([sys.executable, str(PACKAGING / "build.py"), str(self.tmp / "assets"), str(self.tmp / "out"), "--version", "v2.0.0"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = self.tmp / "out"
        self.assertEqual(len(list((out / "pypi").glob("*.whl"))), 4)
        self.assertEqual(len(list((out / "npm").glob("*.tgz"))), 5)
        self.assertTrue((out / "homebrew" / "superfluid.rb").is_file())

    @unittest.skipUnless(shutil.which("npm"), "needs npm")
    def test_build_packages_a_release_without_windows_and_one_platform_alone(self):
        make_all(self.tmp / "assets", "2.0.0", windows=False)
        build = [sys.executable, str(PACKAGING / "build.py"), str(self.tmp / "assets")]
        p = subprocess.run([*build, str(self.tmp / "out"), "--version", "2.0.0"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(sorted(w.name.rsplit("-", 1)[1] for w in (self.tmp / "out" / "pypi").glob("*.whl")),
                         ["macosx_15_0_arm64.whl", "manylinux_2_28_aarch64.whl", "manylinux_2_28_x86_64.whl"])
        self.assertEqual(len(list((self.tmp / "out" / "npm").glob("*.tgz"))), 4)
        # As release.yml tries the Windows packages where the tarball was built.
        make_release(self.tmp / "assets", WINDOWS, "2.0.0")
        p = subprocess.run([*build, str(self.tmp / "one"), "--version", "2.0.0", "--only", "pypi,npm", "--targets", WINDOWS], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([w.name for w in (self.tmp / "one" / "pypi").glob("*.whl")], ["superfluid_cli-2.0.0-py3-none-win_amd64.whl"])
        self.assertEqual(sorted(t.name for t in (self.tmp / "one" / "npm").glob("*.tgz")),
                         ["basecompute-superfluid-2.0.0.tgz", "basecompute-superfluid-win32-x64-2.0.0.tgz"])
        self.assertFalse((self.tmp / "one" / "homebrew").exists())
        p = subprocess.run([*build, str(self.tmp / "bad"), "--version", "2.0.0", "--targets", "x86_64-apple-darwin"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("no such target x86_64-apple-darwin", p.stderr)

    @unittest.skipUnless(shutil.which("npm"), "needs npm")
    def test_build_takes_relative_paths_from_the_working_directory(self):
        # publish.yml runs `build.py assets out --version "$V"` from the checkout.
        # npm pack ran inside the staged package, so a relative out named a
        # folder under it and the first tarball failed with ENOENT.
        make_all(self.tmp / "assets", "2.0.0")
        p = subprocess.run([sys.executable, str(PACKAGING / "build.py"), "assets", "out", "--version", "2.0.0"],
                           cwd=self.tmp, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = self.tmp / "out"
        self.assertEqual(len(list((out / "pypi").glob("*.whl"))), 4)
        self.assertEqual(len(list((out / "npm").glob("*.tgz"))), 5)
        self.assertTrue((out / "homebrew" / "superfluid.rb").is_file())
        # It prints each package under out as given, and each is there.
        printed = p.stdout.splitlines()
        self.assertEqual(len(printed), 10, p.stdout)
        self.assertTrue(all(not Path(line).is_absolute() and (self.tmp / line).is_file() for line in printed), p.stdout)

    def test_build_stops_at_the_first_problem_with_one_line(self):
        make_release(self.tmp / "assets", "aarch64-apple-darwin", "2.0.0")
        p = subprocess.run([sys.executable, str(PACKAGING / "build.py"), str(self.tmp / "assets"), str(self.tmp / "out"), "--version", "2.0.1", "--only", "pypi"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stderr, "build.py: aarch64-apple-darwin: the release is superfluid 2.0.0, not 2.0.1\n")


if __name__ == "__main__":
    unittest.main()
