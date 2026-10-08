"""Shared pieces of the competitor suite: an OpenAI-compatible HTTP client
that can time and abort streams, server process lifecycle, and result files.

Standard library only, Python 3.9+, so it runs with whatever Python a box has.
"""
import http.client
import json
import os
import signal
import socket
import statistics
import subprocess
import time
import urllib.parse


# ---------------------------------------------------------------- HTTP client


class Reply:
    """One request's outcome. `status` is the HTTP status (0 when the
    connection failed before one arrived)."""

    def __init__(self):
        self.status = 0
        self.headers = {}
        self.text = ""
        self.reasoning = ""
        self.tool_calls = []
        self.finish = None
        self.usage = None
        self.error = None  # error body / event / exception text
        self.error_kind = None  # "http" | "event" | "transport" | "timeout" | "truncated"
        self.t0 = time.perf_counter()
        self.stamps = []  # seconds since t0 of each content-bearing chunk
        self.wall = None
        self.raw = None

    @property
    def ttft(self):
        return self.stamps[0] if self.stamps else None

    @property
    def max_gap(self):
        g = [b - a for a, b in zip(self.stamps, self.stamps[1:])]
        return max(g) if g else None

    @property
    def ok(self):
        return self.status == 200 and self.error is None

    @property
    def typed_error(self):
        """A failure the client can act on: an HTTP error status with a body,
        or an SSE error event. Transport resets and silent truncation are not."""
        return self.error_kind in ("http", "event")

    def summary(self):
        return {
            "status": self.status, "ok": self.ok, "ttft": self.ttft, "max_gap": self.max_gap,
            "chunks": len(self.stamps), "wall": self.wall, "finish": self.finish,
            "text_len": len(self.text), "tool_calls": len(self.tool_calls), "usage": self.usage,
            "error_kind": self.error_kind, "error": None if self.error is None else str(self.error)[:400],
            "retry_after": self.headers.get("retry-after"),
        }


def _conn(url, timeout):
    u = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path.rstrip("/")


def get_json(url, path, timeout=10):
    c, base = _conn(url, timeout)
    try:
        c.request("GET", base + path)
        r = c.getresponse()
        body = r.read()
        return r.status, (json.loads(body) if body else None)
    finally:
        c.close()


def post_json(url, path, body, headers=None, timeout=600):
    c, base = _conn(url, timeout)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    try:
        c.request("POST", base + path, body=json.dumps(body), headers=h)
        r = c.getresponse()
        raw = r.read()
        try:
            return r.status, json.loads(raw) if raw else None, {k.lower(): v for k, v in r.getheaders()}
        except ValueError:
            return r.status, raw.decode("utf-8", "replace"), {k.lower(): v for k, v in r.getheaders()}
    finally:
        c.close()


def chat(url, body, headers=None, timeout=900, abort_after_chunks=None, abort_after_s=None, on_chunk=None):
    """POST /v1/chat/completions. Streams when body["stream"] is true.

    abort_after_chunks / abort_after_s close the socket mid-stream (a client
    disconnect). For a non-streaming request abort_after_s closes the socket
    before the reply arrives. on_chunk(reply) is called after every
    content-bearing chunk and may return "abort" to disconnect."""
    rep = Reply()
    c, base = _conn(url, timeout if abort_after_s is None or body.get("stream") else abort_after_s)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    try:
        c.request("POST", base + "/v1/chat/completions", body=json.dumps(body), headers=h)
        r = c.getresponse()
        rep.status = r.status
        rep.headers = {k.lower(): v for k, v in r.getheaders()}
        if r.status != 200:
            raw = r.read().decode("utf-8", "replace")
            rep.error, rep.error_kind = raw, "http"
            return rep
        if not body.get("stream"):
            v = json.loads(r.read())
            rep.raw = v
            ch = (v.get("choices") or [{}])[0]
            m = ch.get("message") or {}
            rep.text = m.get("content") or ""
            rep.reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
            rep.tool_calls = m.get("tool_calls") or []
            rep.finish = ch.get("finish_reason")
            rep.usage = v.get("usage")
            rep.stamps = [time.perf_counter() - rep.t0]
            return rep
        calls = {}
        saw_done = False
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                if line.startswith("event:") and "error" in line:
                    rep.error_kind = "event"
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                v = json.loads(data)
            except ValueError:
                continue
            if "error" in v and not v.get("choices"):
                rep.error, rep.error_kind = v["error"], "event"
                continue
            if v.get("usage"):
                rep.usage = v["usage"]
            for ch in v.get("choices") or []:
                d = ch.get("delta") or {}
                bearing = False
                if d.get("content"):
                    rep.text += d["content"]
                    bearing = True
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    rep.reasoning += rc
                    bearing = True
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                    slot["id"] = tc.get("id") or slot["id"]
                    f = tc.get("function") or {}
                    slot["function"]["name"] += f.get("name") or ""
                    slot["function"]["arguments"] += f.get("arguments") or ""
                    bearing = True
                if ch.get("finish_reason"):
                    rep.finish = ch["finish_reason"]
                if bearing:
                    rep.stamps.append(time.perf_counter() - rep.t0)
                    if on_chunk and on_chunk(rep) == "abort":
                        rep.error_kind = rep.error_kind or "aborted"
                        return rep
                    if abort_after_chunks is not None and len(rep.stamps) >= abort_after_chunks:
                        rep.error_kind = "aborted"
                        return rep
                    if abort_after_s is not None and rep.stamps[-1] >= abort_after_s:
                        rep.error_kind = "aborted"
                        return rep
        rep.tool_calls = [calls[k] for k in sorted(calls)]
        if not saw_done and rep.error is None and rep.finish is None:
            rep.error, rep.error_kind = "stream ended without finish_reason or [DONE]", "truncated"
    except socket.timeout as e:
        rep.error, rep.error_kind = repr(e), "timeout"
    except (OSError, http.client.HTTPException) as e:
        rep.error, rep.error_kind = repr(e), "transport"
    except ValueError as e:  # a 200 whose body is not JSON
        rep.error, rep.error_kind = repr(e), "malformed"
    finally:
        rep.wall = time.perf_counter() - rep.t0
        c.close()
    return rep


def run_parallel(fns):
    """Run zero-argument callables on threads; return their results in order."""
    import threading

    out = [None] * len(fns)

    def go(i):
        try:
            out[i] = fns[i]()
        except Exception as e:  # noqa: BLE001
            out[i] = e

    ts = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(len(fns))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


# ------------------------------------------------------------ server process


class Server:
    """One server under test: started in its own process group so stop()
    takes its workers with it."""

    def __init__(self, name, argv, env, port, log_path, cwd=None, probe="/v1/models", model_id=None, worker_pattern=None,
                 after_start=None):
        self.name, self.argv, self.port, self.log_path = name, argv, port, log_path
        self.env = dict(os.environ, **env)
        self.cwd, self.probe = cwd, probe
        self.model_id = model_id
        self.worker_pattern = worker_pattern  # pgrep -f pattern of a separate worker process, if the system has one
        self.after_start = after_start
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"

    def start(self, timeout=600):
        if port_open(self.port):
            raise RuntimeError(f"port {self.port} already in use before starting {self.name}")
        log = open(self.log_path, "ab")
        log.write(f"\n=== start {time.strftime('%H:%M:%S')} {' '.join(self.argv)}\n".encode())
        log.flush()
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.argv, env=self.env, cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        self.wait_ready(timeout)
        if self.after_start:
            self.after_start(self)
        if not self.model_id:
            st, v = get_json(self.url, "/v1/models")
            self.model_id = v["data"][0]["id"]
        return time.perf_counter() - t0

    def wait_ready(self, timeout=600):
        """Ready = the probe answers 200 AND /v1/models lists a model (an
        unknown model id 404s on chat, so readiness must see the id)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited with {self.proc.returncode}; see {self.log_path}")
            try:
                st, v = get_json(self.url, self.probe, timeout=2)
                if st == 200:
                    if self.probe != "/v1/models":
                        return
                    if isinstance(v, dict) and (v.get("data") or v.get("models")):
                        return
            except (OSError, ValueError, http.client.HTTPException):
                pass
            time.sleep(0.25)
        raise RuntimeError(f"{self.name} not ready after {timeout}s; see {self.log_path}")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def children(self):
        if not self.proc:
            return []
        out = subprocess.run(["pgrep", "-g", str(self.proc.pid)], capture_output=True, text=True).stdout.split()
        return [int(p) for p in out if int(p) != self.proc.pid]

    def worker_pids(self):
        if not self.worker_pattern or not self.proc:
            return []
        mine = set(self.children())
        out = subprocess.run(["pgrep", "-f", self.worker_pattern], capture_output=True, text=True).stdout.split()
        return [int(p) for p in out if int(p) in mine]

    def kill_main(self):
        """SIGKILL the server process alone (a crash), then reap its group."""
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        self._kill_group(signal.SIGKILL)

    def _kill_group(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def stop(self, grace=15):
        if not self.proc:
            return
        if self.proc.poll() is None:
            self._kill_group(signal.SIGTERM)
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                pass
        self._kill_group(signal.SIGKILL)
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.time() + 15
        while port_open(self.port) and time.time() < deadline:
            time.sleep(0.2)


def port_open(port):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


# ------------------------------------------------------------------- results


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def r3(x):
    return None if x is None else round(x, 3)


def thermal():
    """The machine's thermal state before and after a cell: `pmset -g therm`
    on macOS; on Linux the GPU's temperature and throttle reasons from
    nvidia-smi, else the hottest thermal zone."""
    if os.uname().sysname == "Darwin":
        try:
            out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return " ".join(l.strip() for l in out.splitlines() if "limit" in l.lower() or "warning" in l.lower()) or out.strip()[:200]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_throttle_reasons.active,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return "gpu " + out.stdout.strip().replace("\n", "; ")
    except (OSError, subprocess.SubprocessError):
        pass
    temps = []
    for z in sorted(os.listdir("/sys/class/thermal")) if os.path.isdir("/sys/class/thermal") else []:
        try:
            temps.append(int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000)
        except (OSError, ValueError):
            pass
    return f"max zone {max(temps):.0f} C" if temps else None
