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
import release  # noqa: E402

TARGET_FORMAT = {
    "aarch64-apple-darwin": ("macho", 0x0100000C),
    "x86_64-unknown-linux-gnu": ("elf", 62),
    "aarch64-unknown-linux-gnu": ("elf", 183),
}


def macho(minos=(15, 0), cpu=0x0100000C) -> bytes:
    """A 64-bit Mach-O header with one LC_BUILD_VERSION load command."""
    build_version = struct.pack("<IIIIII", 0x32, 24, 1, (minos[0] << 16) | (minos[1] << 8), 15 << 16, 0)
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cpu, 0, 2, 1, len(build_version), 0, 0)
    return header + build_version + b"\0" * 64


def elf(machine=62, glibc=(17, 28)) -> bytes:
    """A 64-bit little-endian ELF header and the glibc symbol versions it needs."""
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    return ident + struct.pack("<HH", 2, machine) + b"\0" * 44 + b"".join(b"GLIBC_2.%d\x00" % v for v in glibc)


def binary_for(target: str, **kw) -> bytes:
    fmt, cpu = TARGET_FORMAT[target]
    if fmt == "macho":
        return macho(cpu=cpu, **{k: v for k, v in kw.items() if k == "minos"})
    return elf(machine=cpu, **{k: v for k, v in kw.items() if k == "glibc"})


def make_release(assets: Path, target: str, version: str = "1.2.3", *, files: dict | None = None,
                 extra: list[tarfile.TarInfo] | None = None, sha: str | None = None, **kw) -> Path:
    """Write superfluid-<target>.tar.gz and its .sha256 into assets, laid out as release.yml lays one out."""
    top = f"superfluid-{target}"
    lib = "libsuperfluid_tokenizer_llamacpp." + ("dylib" if target.endswith("darwin") else "so")
    tree = {
        "bin/superfluid": binary_for(target, **kw),
        "bin/superfluid-workerd": binary_for(target, **kw),
        "bin/superfluid-noded": binary_for(target, **kw),
        "libexec/superfluid/superfluid-worker-llamacpp": binary_for(target, **kw),
        f"libexec/superfluid/{lib}": binary_for(target, **kw),
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


def make_all(assets: Path, version: str = "1.2.3") -> None:
    for target in release.TARGETS:
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
            release.target_of(Path("superfluid-x86_64-pc-windows-msvc.tar.gz"))

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

    def stand_in_release(self, release_dir: Path, executable=True) -> None:
        (release_dir / "bin").mkdir(parents=True)
        (release_dir / "libexec" / "superfluid").mkdir(parents=True)
        for command in release.COMMANDS:
            script = release_dir / "bin" / command
            script.write_text(self.STAND_IN)
            script.chmod(0o755 if executable else 0o644)

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

    def test_the_python_launcher_says_how_to_repair_a_release_it_cannot_run(self):
        pkg = self.tmp / "site" / "superfluid_cli"
        shutil.copytree(PACKAGING / "python" / "superfluid_cli", pkg)
        self.stand_in_release(pkg / "release", executable=False)
        p = subprocess.run([sys.executable, "-c", "import superfluid_cli; superfluid_cli.main()"],
                           env={**os.environ, "PYTHONPATH": str(self.tmp / "site")}, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("pip install --force-reinstall superfluid-cli", p.stderr)

    @unittest.skipUnless(shutil.which("node"), "needs node")
    def test_the_npm_launcher_runs_this_platforms_package(self):
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
        p = subprocess.run(["node", str(bin_js), "a b", "--c"], capture_output=True, text=True)
        self.assertEqual((p.returncode, p.stdout), (7, "ran superfluid-noded a b --c\n"), p.stderr)
        shutil.rmtree(platform_pkg)
        p = subprocess.run(["node", str(bin_js)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn(f"@basecompute/superfluid-{npm_platform} is not installed", p.stderr)


@unittest.skipUnless(shutil.which("npm"), "needs npm")
class NpmTest(Scratch):
    def test_four_packages_each_platform_restricted_and_the_main_one_pointing_at_them(self):
        make_all(self.tmp / "assets")
        built = build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3")
        self.assertEqual([p.name for p in built], [
            "basecompute-superfluid-darwin-arm64-1.2.3.tgz",
            "basecompute-superfluid-linux-x64-1.2.3.tgz",
            "basecompute-superfluid-linux-arm64-1.2.3.tgz",
            "basecompute-superfluid-1.2.3.tgz",
        ])

        def manifest(tgz):
            with tarfile.open(tgz) as tar:
                return json.load(tar.extractfile("package/package.json")), {m.name: m for m in tar}

        linux, members = manifest(built[1])
        self.assertEqual((linux["name"], linux["os"], linux["cpu"], linux["libc"]), ("@basecompute/superfluid-linux-x64", ["linux"], ["x64"], ["glibc"]))
        self.assertEqual(members["package/release/bin/superfluid"].mode & 0o777, 0o755)
        self.assertEqual(members["package/release/compat.json"].mode & 0o777, 0o644)
        for name in ("package/release/installed-by", "package/LICENSE", "package/NOTICE", "package/README.md"):
            self.assertIn(name, members)
        mac, _ = manifest(built[0])
        self.assertNotIn("libc", mac)
        main, members = manifest(built[3])
        self.assertEqual(main["name"], "@basecompute/superfluid")
        self.assertEqual(main["optionalDependencies"], {t["name"]: "1.2.3" for t in build_packages.TARGETS.values()})
        self.assertEqual(main["bin"], {c: f"bin/{c}.js" for c in release.COMMANDS})
        for command in release.COMMANDS:
            self.assertEqual(members[f"package/bin/{command}.js"].mode & 0o777, 0o755)
        self.assertIn("package/lib/launch.js", members)

    def test_a_release_missing_a_platform_makes_no_packages(self):
        make_release(self.tmp / "assets", "aarch64-apple-darwin")
        with self.assertRaisesRegex(release.ReleaseError, "no superfluid-x86_64-unknown-linux-gnu.tar.gz"):
            build_packages.build(self.tmp / "assets", self.tmp / "out", "1.2.3")


class FormulaTest(Scratch):
    def test_the_formula_names_each_tarball_its_checksum_and_the_macos_it_needs(self):
        make_all(self.tmp / "assets")
        rb = formula.build(self.tmp / "assets", self.tmp / "out", "1.2.3").read_text()
        for target in release.TARGETS:
            self.assertIn(f'url "https://github.com/basecompute/superfluid/releases/download/v1.2.3/superfluid-{target}.tar.gz"', rb)
            digest = (self.tmp / "assets" / f"superfluid-{target}.tar.gz.sha256").read_text().split()[0]
            self.assertIn(f'sha256 "{digest}"', rb)
        self.assertIn("depends_on macos: :sequoia", rb)
        self.assertIn('(libexec/"installed-by").write "Homebrew: brew upgrade superfluid\\n"', rb)
        self.assertNotIn("{", rb.replace("#{", ""), "every placeholder was filled")

    def test_a_minimum_between_named_macos_versions_takes_the_next_one(self):
        shas = {t: "0" * 64 for t in release.TARGETS}
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
        self.assertEqual(len(list((out / "pypi").glob("*.whl"))), 3)
        self.assertEqual(len(list((out / "npm").glob("*.tgz"))), 4)
        self.assertTrue((out / "homebrew" / "superfluid.rb").is_file())

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
        self.assertEqual(len(list((out / "pypi").glob("*.whl"))), 3)
        self.assertEqual(len(list((out / "npm").glob("*.tgz"))), 4)
        self.assertTrue((out / "homebrew" / "superfluid.rb").is_file())
        # It prints each package under out as given, and each is there.
        printed = p.stdout.splitlines()
        self.assertEqual(len(printed), 8, p.stdout)
        self.assertTrue(all(not Path(line).is_absolute() and (self.tmp / line).is_file() for line in printed), p.stdout)

    def test_build_stops_at_the_first_problem_with_one_line(self):
        make_release(self.tmp / "assets", "aarch64-apple-darwin", "2.0.0")
        p = subprocess.run([sys.executable, str(PACKAGING / "build.py"), str(self.tmp / "assets"), str(self.tmp / "out"), "--version", "2.0.1", "--only", "pypi"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stderr, "build.py: aarch64-apple-darwin: the release is superfluid 2.0.0, not 2.0.1\n")


if __name__ == "__main__":
    unittest.main()
