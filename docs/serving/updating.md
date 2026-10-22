# Updating superfluid

superfluid updates itself with `superfluid self update`, and `superfluid serve` says once a day when a newer release is out. Releases are kept side by side, so an update never overwrites the copy that is running and the version it replaced stays for going back.

```sh
superfluid self check            # is a newer release out?
superfluid self update           # install the newest release that fits this machine
superfluid self update --dry-run # say what it would do
superfluid self list             # the versions here, which is current, which was replaced
superfluid self use 0.1.3        # go back to a version that is still here
```

## Where releases live

The install script unpacks a release under `$SUPERFLUID_HOME/versions/<version>/` (`~/.superfluid/versions`) and links `superfluid`, `superfluid-workerd` and `superfluid-noded` in `$SUPERFLUID_PREFIX/bin` (`~/.local/bin`) to it through a `current` link:

```text
~/.local/bin/superfluid            -> ~/.superfluid/versions/current/bin/superfluid
~/.local/bin/superfluid-0.1.3      -> ~/.superfluid/versions/0.1.3/bin/superfluid
~/.superfluid/versions/current     -> 0.1.4
~/.superfluid/versions/0.1.4/      bin/ libexec/superfluid/ compat.json
~/.superfluid/versions/0.1.3/
```

The install script places the release it downloaded with that release's own `superfluid self install`, which proves it the way an update does; a release that is refused stops the script, and nothing is copied in its place. Only a release from before `self install` existed is copied into `~/.local/bin` the old way.

Each version finds its workers beside its own real executable, so a server started from 0.1.3 keeps spawning 0.1.3's workers after 0.1.4 became current. The `superfluid-<version>` names run a specific copy.

## What an update does

1. Reads the release listing and picks the newest release for this platform. With a baseRT engine here (an installed `basert` runtime, or `BASERT_LIB`), the release must declare the engine's ABI number, which releases carry in an asset name; with llama.cpp or MLX only, the newest release fits. `--version <v>` names one instead.
2. Downloads the tarball and its `.sha256` into a staging directory and checks the digest, then unpacks with the same rules as a runtime install (no member may land outside the tree).
3. Runs the new copy's `--version` and `runtimes --json`: it must start, find its workers, and, when an engine is in use, report the `basert` runtime ready. A copy that fails is removed and the current version is untouched.
4. Makes it current by replacing the `current` link in one step, and remembers the version it replaced.
5. Removes older versions. Kept: current, the one it replaced, any version with a `keep` file inside its directory, and any version a running superfluid process started from. A version that cannot be removed is said and left for the next update; the update itself has succeeded.

A running server keeps the version it started with; its next start uses the new one. A server started by `superfluid launch` is stopped with `superfluid stop`. One update, install or switch runs at a time: a second waits for the first and then decides on what it left. An installed version is never written to: a release placed again over a working copy is kept as it is, and only a copy that is proved first replaces one that no longer works.

After an update, the runtime installs under `$SUPERFLUID_HOME/runtimes` still carry the worker copied in when they were installed. A newer daemon that does not speak that worker's protocol says so in `superfluid runtimes`, and `superfluid runtime repair <id>` installs the runtime again with the new worker.

## The daily check

After the server is up, `superfluid serve` reads the release listing at most once a day, with a five-second limit, and prints one line when a release newer than the running one fits here. A server that stays up looks again every hour, so it learns of a release published later, or one past its hold-off, when the day is up:

```text
superfluid 0.1.4 is available (this is 0.1.3): `superfluid self update` installs it; SUPERFLUID_NO_UPDATE_CHECK=1 silences this
```

A release is announced a few days after it is published, so early reports land first; `self update` installs it as soon as it is out. The check is skipped with `--offline` or `SUPERFLUID_NO_UPDATE_CHECK=1`, and it never delays serving. What it learned is in `$SUPERFLUID_HOME/update-check.json` and in `superfluid self list`.

## Other ways superfluid was installed

`self update` only touches a superfluid under `versions/`. A copy the install script placed before versions existed (real files in `~/.local/bin`) is kept under `versions/` as the version it is by its first `self update`, so `self use` can go back to it, and the files in `~/.local/bin` become links; its workers in `~/.local/libexec/superfluid` stay while a server still runs from that copy. A superfluid from Homebrew, `cargo install` or a source build is updated the way it was installed, and the command says which.

`SUPERFLUID_RELEASES` names another release listing, a URL or a file in GitHub's releases JSON shape, for a mirror.
