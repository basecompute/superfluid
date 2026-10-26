// Runs a superfluid command for `npm install -g @basecompute/superfluid`.
//
// Copied into the package as lib/launch.js by packaging/npm/build_packages.py;
// bin/superfluid.js and its siblings call it with their command's name. npm
// installs the release for this machine as one optional dependency
// (@basecompute/superfluid-<platform>), and this runs the command from its
// release/ folder, where superfluid finds its runtime workers beside its own
// bin/ and the installed-by file that leaves updates to npm.
"use strict";

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

// Keep in step with TARGETS in packaging/npm/build_packages.py.
const PACKAGES = {
  "darwin-arm64": "@basecompute/superfluid-darwin-arm64",
  "linux-x64": "@basecompute/superfluid-linux-x64",
  "linux-arm64": "@basecompute/superfluid-linux-arm64",
};

function fail(message) {
  console.error(`superfluid: ${message}`);
  process.exit(1);
}

// The first bytes of a 64-bit Mach-O file and of an ELF file.
function isBinary(file) {
  const head = Buffer.alloc(4);
  const fd = fs.openSync(file, "r");
  try {
    fs.readSync(fd, head, 0, 4, 0);
  } finally {
    fs.closeSync(fd);
  }
  return head.equals(Buffer.from([0xcf, 0xfa, 0xed, 0xfe])) || head.equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46]));
}

// Mark the release's binaries executable again, found by header, for a
// package manager that dropped the bits when unpacking.
function restoreModes(release) {
  for (const dir of [path.join(release, "bin"), path.join(release, "libexec", "superfluid")]) {
    for (const name of fs.readdirSync(dir)) {
      const file = path.join(dir, name);
      if (fs.statSync(file).isFile() && isBinary(file)) fs.chmodSync(file, 0o755);
    }
  }
}

module.exports = function launch(command) {
  const platform = `${process.platform}-${process.arch}`;
  const pkg = PACKAGES[platform];
  if (!pkg) {
    fail(`no release for ${platform}; superfluid ships for macOS on Apple silicon and Linux on x64 and arm64 (https://superfluid.sh)`);
  }
  let release;
  try {
    release = path.join(path.dirname(require.resolve(`${pkg}/package.json`)), "release");
  } catch {
    fail(`${pkg} is not installed; install @basecompute/superfluid again without --omit=optional or --no-optional`);
  }
  const exe = path.join(release, "bin", command);
  try {
    fs.accessSync(exe, fs.constants.X_OK);
  } catch {
    try {
      restoreModes(release);
    } catch (e) {
      fail(`${exe} is not executable and could not be made so (${e.message}); install @basecompute/superfluid again`);
    }
  }

  const child = spawn(exe, process.argv.slice(2), { stdio: "inherit" });
  // A terminal's Ctrl-C and Ctrl-\ reach the whole foreground group, child
  // included, so this process only outlives them; a signal sent to this
  // process alone (a supervisor's SIGTERM) is passed on.
  for (const signal of ["SIGINT", "SIGQUIT"]) process.on(signal, () => {});
  for (const signal of ["SIGTERM", "SIGHUP"]) process.on(signal, () => child.kill(signal));
  child.on("error", (e) => fail(`could not start ${exe}: ${e.message}`));
  child.on("exit", (code, signal) => {
    if (signal) {
      process.removeAllListeners(signal);
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 1);
    }
  });
};
