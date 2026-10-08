"""superfluid_mlx: the Python side of superfluid-adapter-mlx.

One `Runtime` per loaded model, driven synchronously from Rust through the
executor's primitive names. Sequences are `mlx-lm` prompt caches (one cache
object per layer, from `make_prompt_cache`); the executor owns lanes, time
and every policy. Refusals travel as `RuntimeError("superfluid:<kind>: ...")`
and the host maps the kind to its typed error.

MLX arrays are values: an in-place update (`a[i] = x`) is visible only
through the same Python object, never through a slice or another array, so
a copy of a cache is independent as soon as each array is materialized
(`mx.contiguous`). That is what makes `seq_copy` safe to append to on both
sides.

Decode rounds are batched: when one `step` feeds several sequences one
token each, their caches are merged into mlx-lm's batch caches (left
padded, masked) and one forward pass serves the round. The merged batch
stays live across rounds that feed the same sequences and is materialized
back into per-sequence caches (`extract`) before anything else touches
them, so every other primitive sees exactly the state a per-sequence
decode would have left.

Memory stays where the executor's ledger can see it. A merged batch is the
only copy of its members' state (their own caches are dropped, not kept
stale beside it), and since it pads every row to its longest member, it is
merged only while that padding fits the pool the ledger was granted: the
longest members decode on their own otherwise (`_batch_group`). A cut or a
partial copy is re-materialized at its new
length, since mlx-lm's `trim` only moves an offset and would keep the whole
buffer alive. MLX keeps freed buffers in an allocator cache for reuse, and
under an agent workload (caches growing, merging, being copied and freed)
that cache grew past 20 GB on a 4B model; it is bounded at load and
emptied whenever a primitive releases state.

A greedy round runs one step ahead, as mlx-lm's own generate loop does. The
token a round answers with is the next round's input, so that next pass is
built on the still-lazy ids and handed to the device (`mx.async_eval`)
before the answer is read back: the device computes token n+1 while the
host is told token n. On Qwen3-4B (4 bit, M1 Max) a lone lane decodes at
104 tok/s this way and at 76 with each round read back before the next is
built. The step ahead is state the executor has not asked for, so it is
undone (`_ahead_drop`: one token trimmed) before anything but its own
continuation touches those sequences, and `lens` never counts it.

A sampled round is drawn on the device too (`step_draw`), by the host
sampler's rules and at the draw the executor names for each lane's
position, so it runs a step ahead like a greedy one: the row never crosses
to the host, and the device is not left idle while the host samples (a
lone sampled lane ran at 74 to 86 tok/s with the row copied out, its speed
set by how the device's clock took the pauses).
"""

import importlib.metadata
import importlib.util
import json
import sys
import types
from pathlib import Path


def _mlx_lm_without_tokenizers():
    """Make `mlx_lm` importable without its package `__init__` and with a
    tokenizer module that loads nothing: the worker runs the model and the
    daemon tokenizes, and `mlx_lm`'s own imports (transformers, through its
    tokenizer utilities) are over half a second of every worker start."""
    if "mlx_lm" in sys.modules:
        return
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.submodule_search_locations:
        return
    pkg = types.ModuleType("mlx_lm")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__spec__ = spec
    pkg.__file__ = spec.origin
    try:
        pkg.__version__ = importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        pkg.__version__ = "unknown"
    sys.modules["mlx_lm"] = pkg
    tokenizers = types.ModuleType("mlx_lm.tokenizer_utils")

    class TokenizerWrapper:
        pass

    def load(*_args, **_kwargs):
        raise RuntimeError("the MLX worker loads no tokenizer; the daemon tokenizes")

    tokenizers.TokenizerWrapper = TokenizerWrapper
    tokenizers.load = load
    sys.modules["mlx_lm.tokenizer_utils"] = tokenizers


_mlx_lm_without_tokenizers()

import mlx.core as mx
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import cache as _cache

MAGIC = b"BRTMLX1\x00"

CACHE_LIMIT_BYTES = 1 << 30

COMPACT_MIN_TOKENS = 256

def _free_bytes():
    """Device memory still free for this process: the GPU's recommended
    working set less what MLX holds (the weights, once loaded); 0 when the
    device does not say."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return max(0, int(info["max_recommended_working_set_size"]) - int(mx.get_active_memory()))
    except Exception:
        return 0

def _refuse(kind, detail=""):
    raise RuntimeError(f"superfluid:{kind}: {detail}" if detail else f"superfluid:{kind}")

def _load_model(path):
    from mlx_lm.utils import load_model

    model, _config = load_model(Path(path), lazy=False)
    return model

def _dtype_name(dt):
    return str(dt).split(".")[-1]

def _arr_bytes(a):
    """(dtype name, shape, raw bytes); bfloat16 travels as its bit pattern."""
    name = _dtype_name(a.dtype)
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return name, list(a.shape), np.ascontiguousarray(np.asarray(a)).tobytes()

def _arr_from(name, shape, raw):
    if name == "bfloat16":
        return mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)
    return mx.array(np.frombuffer(raw, dtype=getattr(np, name)).reshape(shape))

def _meta_out(m):
    """A cache's `meta_state` as JSON: a string, or a tuple or list of them,
    nested as deep as the cache nests (a `CacheList` carries its members'
    class names and their own meta states)."""
    if isinstance(m, tuple):
        return {"kind": "tuple", "v": [_meta_out(x) for x in m]}
    if isinstance(m, list):
        return {"kind": "list", "v": [_meta_out(x) for x in m]}
    return {"kind": "str", "v": "" if m is None else str(m)}

def _meta_in(m):
    """`_meta_out`, read back. An item that is not an object is a string of
    the flat form earlier payloads were written in."""
    if not isinstance(m, dict):
        return m
    if m["kind"] == "tuple":
        return tuple(_meta_in(x) for x in m["v"])
    if m["kind"] == "list":
        return [_meta_in(x) for x in m["v"]]
    return m["v"]

def _arg_sources(model):
    """Getters over the model args; then a child module's resolved args
    (`language_model` in the multimodal layout that nests the text model);
    then the raw `text_config` dict."""
    args = model.args
    sources = [args]
    for name, child in model.children().items():
        if hasattr(child, "args"):
            sources.append(child.args)
    sources.append(getattr(args, "text_config", None))
    for src in sources:
        if src is None:
            continue
        if isinstance(src, dict):
            yield src.get
        else:
            def get(k, d=None, _s=src):
                return getattr(_s, k, d)
            yield get

def _text_arg(model, key):
    for get in _arg_sources(model):
        v = get(key)
        if v is not None:
            return v
    raise AttributeError(f"the model args carry no {key}")

def _attn_dims(model):
    """(kv heads, head dim) from the first args that name them; (0, 0) for a
    model without attention dimensions."""
    for get in _arg_sources(model):
        n_heads = get("num_attention_heads") or 0
        if not n_heads:
            continue
        n_kv = get("num_key_value_heads") or n_heads
        head_dim = get("head_dim") or ((get("hidden_size") or 0) // n_heads)
        return int(n_kv), int(head_dim)
    return 0, 0

def _holds_kv(c):
    """A layer cache that stores keys and values for every token (attention),
    as against a recurrent layer's state, which is one for the sequence."""
    subs = getattr(c, "caches", None)
    if isinstance(subs, (tuple, list)):
        return any(_holds_kv(s) for s in subs)
    return hasattr(c, "keys") and hasattr(c, "values")

def _act_itemsize(model):
    """Bytes of one element of the model's hidden states, which its caches
    keep keys and values in: the embedding's type (its scales' on a quantized
    embedding), else the first floating parameter's. Read from the
    parameters' types alone, so a lazily loaded model reads nothing."""
    floating = [(k, a) for k, a in tree_flatten(model.parameters()) if mx.issubdtype(a.dtype, mx.floating)]
    for k, a in floating:
        if "embed" in k:
            return int(a.dtype.size)
    return int(floating[0][1].dtype.size) if floating else 2

def _kv_bytes_per_token(model, template):
    """Bytes one token leaves in a sequence's caches, over the layers whose
    cache holds keys and values for every token: the pool's unit. A
    recurrent layer's state is one for the sequence, not per token, and is
    not counted. Counted on every layer, a hybrid's cell was priced at 32768
    bytes on LFM2-1.2B (6 of its 16 layers attend) where its caches grow by
    12288 a token. 0 when the model names no attention dimensions."""
    n_kv, head_dim = _attn_dims(model)
    attending = sum(1 for c in template if _holds_kv(c))
    return attending * 2 * int(n_kv) * int(head_dim) * _act_itemsize(model)

def sizing(path):
    """What sizing a window for the model at `path` takes, without loading
    its weights: (a token's KV bytes, the GPU's recommended working set).
    The model is built lazily, as mlx-lm builds it, so the price comes from
    the same caches a load makes; nothing is read past the headers."""
    from mlx_lm.utils import load_model

    model, _config = load_model(Path(path), lazy=True)
    price = _kv_bytes_per_token(model, _cache.make_prompt_cache(model))
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return int(price), int(info["max_recommended_working_set_size"])

def _is_rotating(c):
    """A sliding-window cache: keeps a window, not the whole sequence."""
    return isinstance(c, getattr(_cache, "RotatingKVCache", ())) or hasattr(c, "max_size")

def _clone(c):
    """An independent copy of one layer cache with fresh arrays."""
    state = tree_unflatten([(k, mx.contiguous(v)) for k, v in tree_flatten(c.state)])
    return type(c).from_state(state, c.meta_state)

def _held_bytes(c):
    """Bytes of buffer one layer cache keeps: its live state and any room
    past it (a cut only moves the offset). A class that does not say holds
    its live state."""
    subs = getattr(c, "caches", None)
    if isinstance(subs, (tuple, list)):
        return sum(_held_bytes(s) for s in subs)
    try:
        return int(c.nbytes)
    except NotImplementedError:
        return _live_bytes(c)

def _live_bytes(c):
    """Bytes of one layer cache's live state (`state` covers only it)."""
    return sum(int(a.nbytes) for _, a in tree_flatten(c.state) if hasattr(a, "nbytes"))

def _compact(caches):
    """`caches` re-materialized at their current length: a layer's `state`
    covers only its live tokens, so the copy drops the buffer past them."""
    out = [_clone(c) for c in caches]
    mx.eval([c.state for c in out])
    return out

class Runtime:
    def __init__(self, path, max_batch, max_seq_len, prefill_chunk=2048, cache_limit=CACHE_LIMIT_BYTES):
        mx.set_cache_limit(int(cache_limit))
        self.model = _load_model(path)
        self.max_batch = int(max_batch)
        self.max_seq_len = int(max_seq_len)
        self.prefill_chunk = max(1, int(prefill_chunk))
        self.seqs = {}
        self.lens = {}
        self._next = 1
        template = _cache.make_prompt_cache(self.model)
        self.trimmable = bool(_cache.can_trim_prompt_cache(template)) and not any(
            _is_rotating(c) for c in template
        )
        self.n_layers = len(template)
        self.batchable = all(hasattr(c, "merge") for c in template)
        self.batch = None
        self.ahead = None
        self.lone_pass = False
        self.pool_cells = 0
        self.kv_bytes_priced = _kv_bytes_per_token(self.model, template)
        measured = self._measure_kv_bytes(kv_only=True) if any(_holds_kv(c) for c in template) else 0
        self.kv_bytes_per_token = max(self.kv_bytes_priced, measured)
        if self.kv_bytes_per_token == 0 and any(hasattr(c, "keys") for c in template):
            self.kv_bytes_per_token = self._measure_kv_bytes()
        self.vocab = int(_text_arg(self.model, "vocab_size"))

    def _measure_kv_bytes(self, kv_only=False):
        """Bytes of cache state one token leaves behind, over every layer, or
        over the layers whose cache holds keys and values (`kv_only`)."""
        caches = _cache.make_prompt_cache(self.model)
        mx.eval(self.model(mx.array([[0]], dtype=mx.int32), cache=caches))
        total = 0
        for c in caches:
            if kv_only and not _holds_kv(c):
                continue
            state = c.state
            arrays = state if isinstance(state, (list, tuple)) else [state]
            for a in arrays:
                if a is not None and hasattr(a, "nbytes"):
                    total += int(a.nbytes)
        return total

    def describe(self):
        return {
            "vocab_size": self.vocab,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_bytes_priced": self.kv_bytes_priced,
            "truncate_partial": self.trimmable,
            "recurrent": not self.trimmable,
            "runtime_version": f"mlx {mx.__version__} mlx-lm {mlx_lm.__version__}",
            "n_layers": self.n_layers,
            "free_bytes": _free_bytes(),
        }

    def mem(self):
        if self.ahead is None:
            mx.synchronize()
        return int(mx.get_active_memory()), int(sum(self.lens.values()))

    def merged(self):
        """How many sequences the live decode batch holds (0 with none)."""
        return 0 if self.batch is None else len(self.batch[0])

    def set_pool(self, cells):
        """The token capacity the executor's ledger was granted (0 for a
        ledger with none): what a merged batch's padding must fit in."""
        self.pool_cells = int(cells)

    def _pick(self, last, draw):
        """One token id per row of `last` ([rows, vocab], f32), lazy: the
        lowest-index argmax, or with `draw` = (spec, us) a draw.

        The draw follows the host sampler (`superfluid-executor`'s `sample`):
        probabilities are the softmax of the whole row at the temperature;
        top-k keeps the k most probable; min-p and top-p cut on those
        probabilities as they are (a top-k cut does not move the top-p
        threshold); the survivors, in descending order, are walked to
        `u` of their mass, `us` holding each row's `u`. Rows that tie are
        ordered as the device sorts them."""
        if draw is None:
            return mx.argmax(last, axis=-1)
        (temperature, top_k, top_p, min_p), us = draw
        probs = mx.softmax(last * (1.0 / temperature), axis=-1)
        if 0 < top_k < self.vocab:
            idx = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[:, :top_k]
            p = mx.take_along_axis(probs, idx, axis=-1)
            order = mx.argsort(-p, axis=-1)
            idx = mx.take_along_axis(idx, order, axis=-1)
            p = mx.take_along_axis(p, order, axis=-1)
        else:
            idx = mx.argsort(-probs, axis=-1)
            p = mx.take_along_axis(probs, idx, axis=-1)
        first = mx.arange(p.shape[1])[None] == 0
        keep = first
        rest = mx.ones_like(first)
        if min_p > 0.0:
            rest = rest & (p >= min_p * p[:, :1])
        if top_p < 1.0:
            rest = rest & ((mx.cumsum(p, axis=-1) - p) < top_p)
        keep = keep | rest
        w = mx.where(keep, p, 0.0)
        mass = mx.cumsum(w, axis=-1)
        target = mx.array(us, dtype=mx.float32)[:, None] * mass[:, -1:]
        hit = (mass > target).astype(mx.int32)
        last_live = mx.argmax(mx.cumsum((w > 0).astype(mx.int32), axis=-1), axis=-1)
        choice = mx.where(mx.max(hit, axis=-1) > 0, mx.argmax(hit, axis=-1), last_live)
        return mx.take_along_axis(idx, choice[:, None], axis=-1)[:, 0]

    def _ids(self, caches, x, draw):
        """`_pick` over the last position of one forward pass of `x`
        ([rows, tokens]) into `caches`, lazy."""
        logits = self.model(x, cache=caches)
        last = logits[:, -1].astype(mx.float32)
        if last.shape[1] != self.vocab:
            _refuse("fatal", f"logits width {last.shape[1]} != vocab {self.vocab}")
        return self._pick(last, draw)

    def _answer(self, sids, caches, ids, then):
        """Answer a round with `ids` (lazy, one per row of `sids`), after
        starting the pass that feeds them back; `then` is how that pass
        picks (None for the argmax, else (spec, us) for the next
        position's draw). `lens` already counts the tokens this round was
        fed."""
        ahead = self.lone_pass and self.trimmable and all(self.lens[sid] + 1 <= self.max_seq_len for sid in sids)
        if not ahead:
            self.ahead = None
            return [int(t) for t in ids.tolist()]
        nxt = self._ids(caches, ids.reshape(-1, 1), then)
        mx.async_eval(nxt)
        fed = [int(t) for t in ids.tolist()]
        self.ahead = {"sids": tuple(sids), "caches": caches, "fed": fed, "next": nxt, "draw": then}
        return fed

    def _ahead_drop(self):
        """Undo the step ahead, if one is out: its token leaves the caches,
        which are then exactly what the executor's own rounds left."""
        a, self.ahead = self.ahead, None
        if a is None:
            return
        mx.eval(a["next"])
        for c in a["caches"]:
            got = c.trim(1)
            if got != 1:
                _refuse("fatal", f"a layer trimmed {got} of 1 token ({type(c).__name__})")

    def _ahead_take(self, feeds, picks, draw, then):
        """Serve `feeds` from the step ahead when they are its continuation:
        the same sequences, each fed the token the last round answered with,
        and picked the way the step ahead picked (the argmax, or the same
        draws). A round that wants every token is answered and steps ahead
        again (`then`: how); one that wants none (a publishing retire's last
        token) finds its token already in. Answers None, with the step
        ahead undone, otherwise."""
        a = self.ahead
        if a is None:
            return None
        wants = [w for _, _, w in feeds]
        same = (
            tuple(sid for sid, _, _ in feeds) == a["sids"]
            and all(len(t) == 1 and t[0] == f for (_, t, _), f in zip(feeds, a["fed"]))
            and ((picks and all(wants) and a["draw"] == draw) or not any(wants))
        )
        if not same:
            self._ahead_drop()
            return None
        for sid in a["sids"]:
            self.lens[sid] += 1
        if not any(wants):
            self.ahead = None
            mx.eval(a["next"])
            return [] if picks else b""
        return self._answer(a["sids"], a["caches"], a["next"], then)

    def _own(self, sid):
        """A sequence's own caches for a pass of its own. If the live decode
        batch holds it, only its row leaves the batch (`_detach`): the other
        lanes stay merged across the round."""
        self._detach(sid)
        try:
            return self.seqs[sid]
        except KeyError:
            _refuse("unknown_seq", str(sid))

    def _get(self, sid):
        """A sequence's own caches, newest state: the batch holds that for
        its members, so a member is taken out of it first."""
        return self._own(sid)

    def _detach(self, sid):
        """Take `sid` out of the live decode batch, if it is there: its row
        becomes its own caches and the batch keeps the rest merged. Copying
        one row and filtering the others out is half of materializing every
        member and merging them back on the next round."""
        if self.batch is None or sid not in self.batch[0]:
            return
        live, caches = self.batch
        keep = [i for i, s in enumerate(live) if s != sid]
        if not keep or not all(hasattr(c, "filter") for c in caches):
            self._flush()
            return
        at = live.index(sid)
        self.seqs[sid] = [_clone(c.extract(at)) for c in caches]
        mx.eval([c.state for c in self.seqs[sid]])
        for c in caches:
            c.filter(keep)
        mx.eval([c.state for c in caches])
        self.batch = (tuple(live[i] for i in keep), caches)
        mx.clear_cache()

    def _flush(self):
        """Materialize the live decode batch into per-sequence caches."""
        if self.batch is None:
            return
        sids, caches = self.batch
        self.batch = None
        out = []
        for i, sid in enumerate(sids):
            if sid in self.seqs:
                self.seqs[sid] = [_clone(c.extract(i)) for c in caches]
                out.extend(c.state for c in self.seqs[sid])
        mx.eval(out)
        del caches, out
        mx.clear_cache()

    def seq_create(self):
        sid = self._next
        self._next += 1
        self.seqs[sid] = _cache.make_prompt_cache(self.model)
        self.lens[sid] = 0
        return sid

    def seq_free(self, sid):
        self._ahead_drop()
        self.seqs.pop(sid, None)
        self.lens.pop(sid, None)
        if self.batch is not None and sid in self.batch[0]:
            live, caches = self.batch
            keep = [i for i, s in enumerate(live) if s != sid]
            if keep and all(hasattr(c, "filter") for c in caches):
                for c in caches:
                    c.filter(keep)
                mx.eval([c.state for c in caches])
                self.batch = (tuple(live[i] for i in keep), caches)
            else:
                self._flush()
        mx.clear_cache()

    def seq_len(self, sid):
        if sid not in self.seqs:
            _refuse("unknown_seq", str(sid))
        return self.lens[sid]

    def _check_cut(self, cur, n):
        if n > cur or (not self.trimmable and n not in (0, cur)):
            _refuse("out_of_boundary", f"cut to {n} of {cur} (trimmable={self.trimmable})")

    def _cut(self, caches, cur, n):
        """`caches` (owned) cut to `n` tokens of state. Every layer must
        remove exactly what was asked (`trim` reports what it could): a
        shortfall would leave the state at a different position than the
        one recorded, so it is a fault, not a rounding."""
        if n == cur:
            return caches
        if n == 0:
            return _cache.make_prompt_cache(self.model)
        want = cur - n
        for c in caches:
            got = c.trim(want)
            if got != want:
                _refuse("fatal", f"a layer trimmed {got} of {want} tokens ({type(c).__name__})")
        return caches

    def _slack_tokens(self, caches):
        """How many tokens of attention state the buffers of `caches` have
        room for past their live state: what earlier cuts left behind."""
        if not self.kv_bytes_per_token:
            return 0
        slack = sum(_held_bytes(c) - _live_bytes(c) for c in caches if not c.empty())
        return max(0, slack) // self.kv_bytes_per_token

    def seq_copy(self, src, dst, n):
        self._ahead_drop()
        s = self._get(src)
        self._get(dst)
        cur = self.lens[src]
        self._check_cut(cur, n)
        if n == 0:
            new = _cache.make_prompt_cache(self.model)
        else:
            new = self._cut([_clone(c) for c in s], cur, n)
            if self._slack_tokens(new) >= COMPACT_MIN_TOKENS:
                new = _compact(new)
        self.seqs[dst] = new
        self.lens[dst] = n
        mx.clear_cache()

    def seq_truncate(self, sid, n):
        self._ahead_drop()
        s = self._get(sid)
        cur = self.lens[sid]
        self._check_cut(cur, n)
        if n == cur:
            return
        cut = self._cut(s, cur, n)
        if n > 0 and self._slack_tokens(cut) >= COMPACT_MIN_TOKENS:
            cut = _compact(cut)
        self.seqs[sid] = cut
        self.lens[sid] = n
        mx.clear_cache()

    def seq_export(self, sid, n):
        self._ahead_drop()
        s = self._get(sid)
        cur = self.lens[sid]
        if n > cur or (not self.trimmable and n != cur):
            _refuse("out_of_boundary", f"export {n} of {cur} (trimmable={self.trimmable})")
        caches = s if n == cur else self._cut([_clone(c) for c in s], cur, n)
        layers, blobs = [], []
        for c in caches:
            layer = {"cls": type(c).__name__, "meta": _meta_out(c.meta_state), "arrays": None}
            if n > 0 and not c.empty():
                arrs = []
                for k, v in tree_flatten(c.state):
                    name, shape, raw = _arr_bytes(v)
                    arrs.append({"k": k, "dt": name, "shape": shape, "len": len(raw)})
                    blobs.append(raw)
                layer["arrays"] = arrs
            layers.append(layer)
        header = json.dumps({"len": int(n), "layers": layers}).encode()
        return MAGIC + len(header).to_bytes(4, "little") + header + b"".join(blobs)

    def seq_import(self, sid, payload):
        self._ahead_drop()
        self._get(sid)
        if self.lens[sid] != 0:
            _refuse("out_of_boundary", "import needs an empty sequence")
        payload = bytes(payload)
        if payload[:8] != MAGIC:
            _refuse("fatal", "not an MLX state payload")
        hl = int.from_bytes(payload[8:12], "little")
        header = json.loads(payload[12 : 12 + hl])
        off = 12 + hl
        n = int(header["len"])
        if n == 0:
            self.seqs[sid] = _cache.make_prompt_cache(self.model)
            self.lens[sid] = 0
            return 0
        if len(header["layers"]) != self.n_layers:
            _refuse("fatal", f"{len(header['layers'])} layers in the payload, model has {self.n_layers}")
        caches = []
        for layer in header["layers"]:
            cls = getattr(_cache, layer["cls"])
            if layer["arrays"] is None:
                _refuse("fatal", "empty layer in a non-empty payload")
            flat = []
            for a in layer["arrays"]:
                raw = payload[off : off + a["len"]]
                off += a["len"]
                flat.append((a["k"], _arr_from(a["dt"], a["shape"], raw)))
            caches.append(cls.from_state(tree_unflatten(flat), _meta_in(layer["meta"])))
        self.seqs[sid] = caches
        self.lens[sid] = n
        return n

    def _batch_group(self, sids):
        """The sequences of one decode round that go through as one merged
        batch, in feed order: all of `sids` while the batch fits the pool,
        else all but the longest.

        A merged batch pads every row to its longest member, so it holds
        `rows x longest` tokens of state where the ledger counts the
        members' own lengths: one 24k-token lane beside three 1k ones is
        96k tokens of buffer for 27k of state, and the ledger would go on
        admitting into memory that is no longer there. The pool is what the
        ledger was granted, so a batch is merged only while its padded size,
        with everything else resident, fits the pool; the longest members
        are left out one by one until it does, and decode on their own,
        unpadded."""
        if not self.pool_cells:
            return sids
        group = sorted(sids, key=lambda sid: self.lens[sid])
        resident = sum(self.lens.values())
        while len(group) >= 2:
            live = sum(self.lens[sid] for sid in group)
            padded = len(group) * self.lens[group[-1]]
            if resident - live + padded <= self.pool_cells:
                break
            group.pop()
        if len(group) < 2:
            return ()
        members = set(group)
        return tuple(sid for sid in sids if sid in members)

    def _step_batched(self, sids, feeds, picks=False, draw=None, then=None):
        """One decode round for several sequences, one token each, as one
        forward pass over merged caches. The merged batch is reused while
        the same sequences are fed in the same order; a subset keeps the
        survivors' rows and hands the others their state back. Answers
        with one item per wanting feed, in feed order: its row's bytes, or
        with `picks` its token id, picked on the device (`_pick`)."""
        for sid in sids:
            if sid not in self.seqs:
                _refuse("unknown_seq", str(sid))
            if self.lens[sid] + 1 > self.max_seq_len:
                _refuse("capacity", f"{self.lens[sid] + 1} tokens exceeds max_seq_len {self.max_seq_len}")
        if self.batch is not None and self.batch[0] != sids:
            live, caches = self.batch
            keep = [live.index(sid) for sid in sids if sid in live]
            joining = sids[len(keep):]
            if (
                joining
                and keep
                and keep == sorted(keep)
                and tuple(live[i] for i in keep) == sids[: len(keep)]
                and all(s not in live for s in joining)
                and all(hasattr(c, "filter") and hasattr(c, "extend") for c in caches)
            ):
                # Lanes that join after the batch's own (the executor feeds
                # lanes in admission order): keep the batch, drop the lanes
                # that left, and append the newcomers, rather than
                # materializing every member and merging them all back.
                back = []
                for i in range(len(live)):
                    if i not in keep and live[i] in self.seqs:
                        self.seqs[live[i]] = [_clone(c.extract(i)) for c in caches]
                        back.extend(c.state for c in self.seqs[live[i]])
                mx.eval(back)
                if len(keep) != len(live):
                    for c in caches:
                        c.filter(keep)
                for layer, c in enumerate(caches):
                    members = [self.seqs[sid][layer] for sid in joining]
                    c.extend(members[0].merge(members))
                mx.eval([c.state for c in caches])
                for sid in joining:
                    self.seqs[sid] = None
                self.batch = (sids, caches)
                del back
                mx.clear_cache()
            elif len(keep) == len(sids) and keep == sorted(keep) and all(hasattr(c, "filter") for c in caches):
                dropped = [i for i in range(len(live)) if i not in keep]
                back = []
                for i in dropped:
                    if live[i] in self.seqs:
                        self.seqs[live[i]] = [_clone(c.extract(i)) for c in caches]
                        back.extend(c.state for c in self.seqs[live[i]])
                mx.eval(back)
                for c in caches:
                    c.filter(keep)
                mx.eval([c.state for c in caches])
                self.batch = (sids, caches)
                del back
                mx.clear_cache()
            else:
                self._flush()
        if self.batch is None:
            caches = []
            for layer in range(self.n_layers):
                members = [self.seqs[sid][layer] for sid in sids]
                caches.append(members[0].merge(members))
            self.batch = (sids, caches)
            for sid in sids:
                self.seqs[sid] = None
        caches = self.batch[1]
        x = mx.array([[t[0]] for _, t, _ in feeds], dtype=mx.int32)
        logits = self.model(x, cache=caches)
        for sid in sids:
            self.lens[sid] += 1
        want = [i for i, (_, _, w) in enumerate(feeds) if w]
        if not want:
            mx.eval(logits)
            return []
        last = logits[:, -1].astype(mx.float32)
        if last.shape[1] != self.vocab:
            _refuse("fatal", f"logits width {last.shape[1]} != vocab {self.vocab}")
        if picks:
            ids = self._pick(last, draw)
            if len(want) == len(feeds):
                return self._answer(sids, caches, ids, then)
            mx.eval(ids)
            ids = ids.tolist()
            return [int(ids[i]) for i in want]
        mx.eval(last)
        arr = np.asarray(last)
        return [np.ascontiguousarray(arr[i]).tobytes() for i in want]

    def step_argmax(self, feeds):
        """`step`, answered with each wanting feed's lowest-index argmax (a
        list of ids) instead of its row: the pass is the same, the row
        never leaves the device."""
        return self.step(feeds, argmax=True)

    def step_draw(self, feeds, spec, us, us_next):
        """`step`, answered with a token drawn for each wanting feed (a list
        of ids): `spec` = (temperature, top_k, top_p, min_p), the same for
        every feed; `us` each feed's draw for this position and `us_next`
        its draw for the next one, which the step ahead samples with."""
        spec = tuple(spec)
        return self.step(feeds, draw=(spec, list(us)), then=(spec, list(us_next)))

    def step(self, feeds, argmax=False, draw=None, then=None):
        """feeds: [(sid, tokens, wants_row)] -> the f32 rows of the feeds that
        want one, in feed order, as one bytes object (or their token ids as
        a list: the argmax with `argmax`, a draw with `draw`).

        A round of single tokens goes through as one merged batch, less
        the sequences its padding has no room for (`_batch_group`); those,
        and every feed of any other round, take a pass each."""
        picks = argmax or draw is not None
        taken = self._ahead_take(feeds, picks, draw, then)
        if taken is not None:
            return taken
        at = {sid: i for i, (sid, _, _) in enumerate(feeds)}
        own = lambda d, sids: None if d is None else (d[0], [d[1][at[sid]] for sid in sids])
        group = ()
        if self.batchable and len(feeds) >= 2 and all(len(t) == 1 for _, t, _ in feeds):
            sids = tuple(sid for sid, _, _ in feeds)
            if len(set(sids)) == len(sids):
                group = self._batch_group(sids)
        # A pass of its own (a prompt chunk) leaves the decode batch merged
        # unless it is for one of the batch's members (`_own`): materializing
        # the batch copies every member's state out, and the next round
        # copies it back in.
        members = set(group)
        self.lone_pass = len(members) == len(feeds) if group else len(feeds) == 1
        merged = iter(
            self._step_batched(group, [f for f in feeds if f[0] in members], picks, own(draw, group), own(then, group))
            if group
            else ()
        )
        rows = []
        for sid, tokens, wants_row in feeds:
            if sid in members:
                if wants_row:
                    rows.append(next(merged))
                continue
            row = self._step_one(sid, tokens, wants_row, picks, own(draw, (sid,)), own(then, (sid,)))
            if row is not None:
                rows.append(row)
        return rows if picks else b"".join(rows)

    def step_rows(self, feeds):
        """feeds: [(sid, tokens, wants_rows)] -> for each feed that wants
        rows, one f32 row after every one of its tokens, all as one bytes
        object in feed order: a drafted continuation verified in one pass."""
        self._ahead_drop()
        out = []
        for sid, tokens, wants in feeds:
            caches = self._own(sid)
            n = len(tokens)
            if n == 0:
                continue
            if self.lens[sid] + n > self.max_seq_len:
                _refuse("capacity", f"{self.lens[sid] + n} tokens exceeds max_seq_len {self.max_seq_len}")
            logits = self.model(mx.array([list(tokens)], dtype=mx.int32), cache=caches)
            self.lens[sid] += n
            if not wants:
                mx.eval([c.state for c in caches])
                continue
            rows = logits[0].astype(mx.float32)
            if rows.shape[1] != self.vocab:
                _refuse("fatal", f"logits width {rows.shape[1]} != vocab {self.vocab}")
            mx.eval(rows)
            out.append(np.asarray(rows).tobytes())
        return b"".join(out)

    def _step_one(self, sid, tokens, wants_row, picks, draw=None, then=None):
        """One sequence's own pass over `tokens`, a prefill chunk at a time:
        its row's bytes (or, with `picks`, its token id) when it wants one,
        else None."""
        caches = self._own(sid)
        n = len(tokens)
        if n == 0:
            if wants_row:
                _refuse("out_of_boundary", "a logits row needs at least one token")
            return None
        if self.lens[sid] + n > self.max_seq_len:
            _refuse("capacity", f"{self.lens[sid] + n} tokens exceeds max_seq_len {self.max_seq_len}")
        # The prompt is run for its state alone, as mlx_lm does: evaluating
        # a chunk's logits would compute the vocabulary head for every one
        # of its tokens (a tenth of the work on a 4B model) to keep the
        # last. The token whose row is wanted runs on its own.
        body = tokens[:-1] if wants_row else tokens
        pos = 0
        while pos < len(body):
            chunk = body[pos : pos + self.prefill_chunk]
            self.model(mx.array(chunk, dtype=mx.int32)[None], cache=caches)
            mx.eval([c.state for c in caches])
            pos += len(chunk)
        self.lens[sid] += n
        if not wants_row:
            return None
        logits = self.model(mx.array([tokens[-1:]], dtype=mx.int32), cache=caches)
        row = logits[0, -1].astype(mx.float32)
        if row.shape[0] != self.vocab:
            _refuse("fatal", f"logits width {row.shape[0]} != vocab {self.vocab}")
        if picks:
            return self._answer((sid,), caches, self._pick(row[None], draw), then)[0]
        mx.eval(row)
        return np.asarray(row).tobytes()
