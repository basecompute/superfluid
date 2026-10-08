# Security Policy

## Supported versions

Superfluid is pre-1.0. Security fixes land on `main` and in the latest release.

## Reporting a vulnerability

Report security issues privately. Do **not** open a public issue.

- Open a private advisory through this repository's "Security" tab
  (Security → Advisories → New draft).

Please include a description and its impact, a minimal reproduction, the
affected commit or release, and any mitigation you suggest. We aim to
acknowledge within 5 business days and to triage within 10; disclosure is
coordinated with you.

## Out of scope

- Issues that need local code execution the user already granted (a model
  directory that ships Python the operator chose to trust, a library the
  operator pointed the daemon at).
- Performance or output differences between runtimes (file a regular issue).
- Vulnerabilities in a runtime the daemon installs (llama.cpp, MLX) or in a
  dependency — report those upstream, then tell us so the pinned build can
  move.

## Hardening notes for operators

The HTTP server is meant for trusted networks. Before exposing it:

- Set `--api-key` (or `--key-policy`). Without one, anything that can reach
  the port can load a model from any path on the host
  (`POST /v1/models/load`), unload one, and upload files and audio.
- Bind to a specific interface; do not bind to `0.0.0.0` without an upstream
  auth layer.
- Put a reverse proxy in front for TLS and request-size limits.
- The unix socket (`--socket`) trusts whoever can open it; keep its directory
  private to the daemon's user.
- A fleet node agent (`superfluid-noded`) should listen on an authenticated
  transport (WireGuard or similar) and be given an `--auth-file`; it refuses
  a non-loopback address without one.
- `SUPERFLUID_MLX_TRUST_MODEL_CODE=1` lets an MLX model directory run its own
  Python as the daemon's user. Leave it unset for models you did not write.
