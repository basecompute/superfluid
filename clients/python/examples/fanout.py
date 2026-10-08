"""Fan out: read one long document once, then branch into several questions at the same time.

    python fanout.py --context README.md [--seed 7]

The document goes into one session and is prefilled once. Each question is a fork of
that session: it shares the document by reference, starts warm, and streams beside the
others. Start the server with `--web 127.0.0.1:8460` (see Client for the token).
"""

import argparse
import shutil
import sys
import textwrap
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superfluid_client import Client, Sampling  # noqa: E402

QUESTIONS = [
    "Summarize it in three bullet points for a busy executive.",
    "Explain it to a ten-year-old in two sentences.",
    "What is the single most surprising detail? Quote it.",
    "Write a haiku about it.",
]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


class Pane:
    def __init__(self, title: str):
        self.title, self.text, self.thinking, self.state, self.reply = title, "", "", "waiting", None
        self.started = self.first = None


def render(panes: list, height: int, first: bool) -> None:
    width = max(40, shutil.get_terminal_size().columns - 2)
    out = []
    for p in panes:
        ttft = f" ttft {p.first - p.started:.2f}s" if p.first and p.started else ""
        warm = f" warm {p.reply.warm_prefix} tok" if p.reply else ""
        out.append(f"{BOLD}{p.title[: width - 30]}{RESET}{DIM}  {p.state}{ttft}{warm}{RESET}")
        body, style = (p.text, "") if p.text.strip() else (p.thinking, DIM)
        lines = []
        for para in body.strip().splitlines() or [""]:
            lines += textwrap.wrap(para, width) or [""]
        lines = lines[-height:]
        out += [f"  {style}{line}{RESET}" for line in lines] + [""] * (height - len(lines))
    if not first:
        sys.stdout.write(f"\033[{len(out)}F")
    sys.stdout.write("".join(f"\033[2K{line}\n" for line in out))
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True, help="a text file to read once and ask about")
    ap.add_argument("--question", action="append", help="ask this instead (repeatable)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--lines", type=int, default=4, help="lines of output shown per branch")
    args = ap.parse_args()

    questions = args.question or QUESTIONS
    doc = Path(args.context).read_text()
    sf = Client()

    base = sf.session(Sampling(temperature=0.7, seed=args.seed), system="You answer questions about a document. Be concise.")
    shared = base.user(f"Here is a document:\n\n{doc}\n\nI will ask you about it.")
    base.assistant("I have read it. Ask away.")

    t0 = time.time()
    primer = base.fork()
    primer.user("Reply with OK.")
    primer.generate(1)
    prefill_s = time.time() - t0
    shared_tokens = sf.inspect(base.id)["summary"]["tokens"]
    print(f"{BOLD}session {base.id}{RESET}: {shared_tokens} tokens of document and setup, prefilled once in {prefill_s:.2f}s\n")

    panes = [Pane(q) for q in questions]
    branches = []
    for i, (q, pane) in enumerate(zip(questions, panes)):
        b = base.fork(seed=args.seed + i)
        b.user(q)
        branches.append(b)

    lock = threading.Lock()

    def run(branch, pane):
        pane.started, pane.state = time.time(), "streaming"

        def on_text(chunk):
            with lock:
                if pane.first is None:
                    pane.first = time.time()
                if chunk.kind == "text":
                    pane.text += chunk.text
                elif chunk.kind == "reasoning":
                    pane.thinking += chunk.text

        pane.reply = branch.generate(args.max_tokens, on_text=on_text)
        pane.state = f"done ({pane.reply.tokens} tok, {pane.reply.finish})"

    threads = [threading.Thread(target=run, args=bp, daemon=True) for bp in zip(branches, panes)]
    t1 = time.time()
    for t in threads:
        t.start()
    first = True
    while any(t.is_alive() for t in threads):
        with lock:
            render(panes, args.lines, first)
        first = False
        time.sleep(0.1)
    render(panes, args.lines, first)

    wall = time.time() - t1
    total = sum(p.reply.tokens for p in panes)
    warm = [p.reply.warm_prefix for p in panes]
    reused = sum(1 for w in warm if w >= 0.9 * shared_tokens)
    print(f"\n{len(panes)} branches, {total} tokens in {wall:.1f}s; {reused} of {len(panes)} reused the "
          f"{shared_tokens}-token shared prefix instead of prefilling it again "
          f"(cached tokens per branch: {', '.join(map(str, warm))}).")
    print(f"{DIM}tree: {base.id} -> " + ", ".join(str(b.id) for b in branches) +
          f" (forked after event {shared.id + 1}; resume any of them later with Client().get(id)){RESET}")


if __name__ == "__main__":
    main()
