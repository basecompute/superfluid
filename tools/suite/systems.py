"""The systems under test on one box, and how to start each.

A system is built per scenario with the knobs that scenario needs (`lanes`
concurrent sequences, `ctx` tokens of context per sequence). Every system gets
the same lanes and per-sequence context so KV capacity is comparable; what a
server does beyond that (paging, caching, preemption) is what is measured.
"""
import json
import os
import shutil
import subprocess
import time

from common import Server, get_json, post_json

PORTS = {
    "superfluid-llamacpp": 18601,
    "superfluid-mlx": 18602,
    "llama-server": 18611,
    "mlx-lm-server": 18612,
    "ollama": 18613,
}

# What each system can be asked to do at all. A scenario that needs a hook a
# system lacks is reported "unsupported", with the reason, instead of run.
CAPS = {
    "superfluid-llamacpp": {"worker": "superfluid-worker-llamacpp", "unload": "openai-models-unload", "qos_header": "x-superfluid-qos"},
    "superfluid-mlx": {"worker": "superfluid-worker-mlx", "unload": "openai-models-unload", "qos_header": "x-superfluid-qos"},
    "llama-server": {"worker": None, "unload": None, "qos_header": None},
    "mlx-lm-server": {"worker": None, "unload": None, "qos_header": None},
    "ollama": {"worker": "ollama/.*/llama-server --model", "unload": "ollama-keep-alive-0", "qos_header": None},
}

OLLAMA_MODEL = "qwen3-4b-q4km"  # a box may name its own with "ollama_model"

# Request fields a system needs for the common request to mean the same
# thing: Ollama's OpenAI route ignores chat_template_kwargs, and turns
# thinking off with reasoning_effort "none" (it also ignores `think`).
BODY_EXTRA = {"ollama": {"reasoning_effort": "none"}}


def load_box(path):
    box = json.load(open(path))
    root = os.path.expanduser(box["root"])
    box["root"] = root
    for k in ("superfluid", "superfluid_home", "llama_server", "llama_lib", "mlx_python", "mlx_site", "ollama"):
        if k in box:
            box[k] = os.path.join(root, box[k])
    box["models"] = {k: os.path.join(root, v) for k, v in box["models"].items()}
    box.setdefault("ollama_model", OLLAMA_MODEL)
    # The variable a shared-library path goes in: DYLD_LIBRARY_PATH on macOS,
    # LD_LIBRARY_PATH on Linux.
    box.setdefault("lib_env", "DYLD_LIBRARY_PATH" if os.uname().sysname == "Darwin" else "LD_LIBRARY_PATH")
    # Systems the box describes itself (mistral.rs, oMLX, vLLM, SGLang ...):
    # {"name": {"argv": [...], "env": {...}, "probe": "/v1/models", "port": N}}.
    # argv and env values may use {gguf} {mlx} {hf} {port} {lanes} {ctx}
    # {kv_tokens} {root}.
    for i, (name, spec) in enumerate(sorted(box.get("extra_systems", {}).items())):
        PORTS.setdefault(name, spec.get("port", 18620 + i))
        CAPS.setdefault(name, {"worker": spec.get("worker"), "unload": None, "qos_header": None})
    return box


def known(box, name):
    """Whether this box can start `name` at all."""
    if name in box.get("extra_systems", {}):
        return True
    need = {"superfluid-llamacpp": ("superfluid", "gguf"), "superfluid-mlx": ("superfluid", "mlx"),
            "llama-server": ("llama_server", "gguf"), "mlx-lm-server": ("mlx_python", "mlx"), "ollama": ("ollama", "gguf")}
    if name not in need:
        return False
    binary, model = need[name]
    return binary in box and model in box["models"]


def superfluid_sha(box):
    try:
        return subprocess.run([box["superfluid"], "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except OSError:
        return None


def build(name, box, run_dir, lanes=8, ctx=4096, extra=None, fresh_state=True, defaults=False):
    """Return a Server for `name`. `run_dir` holds its logs and state;
    fresh_state=False keeps the sessions dir (the restart scenario).

    defaults=True starts the system as a user would with no tuning: the
    model and the port the harness listens on, nothing else (lanes and ctx
    are ignored); what its defaults do is the measurement."""
    port = PORTS[name]
    log = os.path.join(run_dir, f"{name}.server.log")
    extra = list(extra or [])
    if name in box.get("extra_systems", {}):
        spec = box["extra_systems"][name]
        fill = {"gguf": box["models"].get("gguf", ""), "mlx": box["models"].get("mlx", ""), "hf": box.get("hf_model", ""),
                "port": port, "lanes": lanes, "ctx": ctx, "kv_tokens": lanes * ctx, "root": box["root"]}
        argv = [str(a).format(**fill) for a in spec["argv"]] + extra
        env = {k: str(v).format(**fill) for k, v in spec.get("env", {}).items()}
        srv = Server(name, argv, env, port, log, probe=spec.get("probe", "/v1/models"), worker_pattern=spec.get("worker"))
        srv.model_id = spec.get("model_id")
        return srv
    if name.startswith("superfluid-") and defaults:
        runtime = name.split("-", 1)[1]
        model = box["models"]["gguf" if runtime == "llamacpp" else "mlx"]
        home = os.path.join(run_dir, f"home-{name}")
        if fresh_state:
            shutil.rmtree(home, ignore_errors=True)
        os.makedirs(home, exist_ok=True)
        argv = [box["superfluid"], "serve", model, "--http", f"127.0.0.1:{port}", "--sessions", os.path.join(home, "sessions"),
                "--socket", f"/tmp/suite-{port}.sock", "--no-tui"] + extra
        srv = Server(name, argv, {"SUPERFLUID_HOME": box["superfluid_home"]}, port, log, probe="/health",
                     worker_pattern=CAPS[name]["worker"])
        srv.model_path = model
        return srv
    if name.startswith("superfluid-"):
        runtime = name.split("-", 1)[1]
        model = box["models"]["gguf" if runtime == "llamacpp" else "mlx"]
        sess = os.path.join(run_dir, f"sess-{name}")
        if fresh_state:
            shutil.rmtree(sess, ignore_errors=True)
        argv = [box["superfluid"], "serve", "--model", model, "--runtime", runtime, "--sessions", sess,
                "--socket", f"/tmp/suite-{port}.sock", "--http", f"127.0.0.1:{port}",
                "--max-batch", str(lanes), "--max-context", str(ctx), "--no-tui"]
        # Speculation off unless the scenario asks for it.
        argv += extra if "--speculate" in extra else ["--speculate", "off"] + extra
        srv = Server(name, argv, {"SUPERFLUID_HOME": box["superfluid_home"]}, port, log, probe="/health",
                     worker_pattern=CAPS[name]["worker"])
        srv.model_path = model
        return srv
    if name == "llama-server" and defaults:
        argv = [box["llama_server"], "-m", box["models"]["gguf"], "--host", "127.0.0.1", "--port", str(port)] + extra
        return Server(name, argv, {box["lib_env"]: box["llama_lib"]}, port, log, probe="/health")
    if name == "llama-server":
        # Unified KV sized lanes x ctx: the same total as everyone else, any
        # one sequence may use up to ctx.
        argv = [box["llama_server"], "-m", box["models"]["gguf"], "-c", str(lanes * ctx), "-np", str(lanes), "-ngl", "99",
                "-kvu", "--kv-unified-per-slot", str(ctx), "--metrics", "--host", "127.0.0.1", "--port", str(port)] + extra
        return Server(name, argv, {box["lib_env"]: box["llama_lib"]}, port, log, probe="/health")
    if name == "mlx-lm-server" and defaults:
        argv = [box["mlx_python"], "-m", "mlx_lm", "server", "--model", box["models"]["mlx"], "--host", "127.0.0.1",
                "--port", str(port)] + extra
        return Server(name, argv, {"PYTHONPATH": box["mlx_site"]}, port, log, probe="/v1/models")
    if name == "mlx-lm-server":
        argv = [box["mlx_python"], "-m", "mlx_lm", "server", "--model", box["models"]["mlx"], "--host", "127.0.0.1",
                "--port", str(port), "--decode-concurrency", str(lanes), "--prompt-concurrency", str(lanes)] + extra
        return Server(name, argv, {"PYTHONPATH": box["mlx_site"]}, port, log, probe="/v1/models")
    if name == "ollama":
        models = os.path.join(box["root"], "ollama-models")
        env = {"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_MODELS": models}
        if not defaults:
            env.update({"OLLAMA_NUM_PARALLEL": str(lanes), "OLLAMA_CONTEXT_LENGTH": str(ctx), "OLLAMA_KEEP_ALIVE": "-1",
                        "OLLAMA_MAX_LOADED_MODELS": "1"})

        name_ = box["ollama_model"]

        def ensure_model(srv):
            st, v = get_json(srv.url, "/api/tags")
            if not any(m.get("name", "").startswith(name_) for m in (v or {}).get("models", [])):
                mf = os.path.join(run_dir, "Modelfile")
                open(mf, "w").write(f"FROM {box['models']['gguf']}\n")
                subprocess.run([box["ollama"], "create", name_, "-f", mf], env=srv.env, check=True,
                               capture_output=True, timeout=1800)
            # Load it now so the first measured request does not pay the load.
            post_json(srv.url, "/api/generate", {"model": name_, "prompt": "", "keep_alive": -1}, timeout=600)
            srv.model_id = name_

        return Server(name, [box["ollama"], "serve"], env, port, log, probe="/api/version",
                      worker_pattern=CAPS["ollama"]["worker"], after_start=ensure_model)
    raise KeyError(name)


def unload(srv):
    """Unload the model through the system's own API. Returns (status, body)."""
    how = CAPS[srv.name]["unload"]
    if how == "openai-models-unload":
        st, v, _ = post_json(srv.url, "/v1/models/unload", {"model": srv.model_id}, timeout=120)
        return st, v
    if how == "ollama-keep-alive-0":
        st, v, _ = post_json(srv.url, "/api/generate", {"model": srv.model_id, "prompt": "", "keep_alive": 0}, timeout=120)
        return st, v
    return None, "no unload API"


def reload(srv):
    how = CAPS[srv.name]["unload"]
    if how == "openai-models-unload":
        st, v, _ = post_json(srv.url, "/v1/models/load", {"model": srv.model_id}, timeout=600)
        if st != 200 and getattr(srv, "model_path", None):
            # A model given with --model is not "known" once unloaded; only
            # its path loads it again, and then it answers under that path.
            st, v, _ = post_json(srv.url, "/v1/models/load", {"id": srv.model_path, "runtime": srv.name.split("-", 1)[1]}, timeout=600)
            if st == 200 and isinstance(v, dict) and v.get("id"):
                srv.model_id = v["id"]
        return st, v
    if how == "ollama-keep-alive-0":
        st, v, _ = post_json(srv.url, "/api/generate", {"model": srv.model_id, "prompt": "", "keep_alive": -1}, timeout=600)
        return st, v
    return None, "no load API"


def versions(box):
    v = dict(box.get("versions", {}))
    v["superfluid"] = superfluid_sha(box)
    v["captured"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return v
