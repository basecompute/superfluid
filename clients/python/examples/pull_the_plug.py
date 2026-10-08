"""Pull the plug: kill -9 the server mid-sentence, restart it, and the reply carries on.

    python pull_the_plug.py <model> [--superfluid PATH] [--after 80]

Starts its own server on a scratch sessions directory, streams a story, SIGKILLs the
server partway, starts it again and continues the same session. Every committed token
survives: the story resumes from the last one, mid-sentence, with nothing repeated.
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superfluid_client import Client, Sampling  # noqa: E402

DIM, BOLD, RED, GREEN, RESET = "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[0m"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, binary: str, model: str, root: Path):
        self.binary, self.model, self.root = binary, model, root
        self.port = free_port()
        self.proc = None

    def start(self) -> Client:
        sessions = self.root / "sessions"
        self.proc = subprocess.Popen(
            [
                self.binary, "serve", self.model,
                "--sessions", str(sessions),
                "--socket", str(self.root / "sf.sock"),
                "--web", f"127.0.0.1:{self.port}",
                "--no-http", "--no-tui",
            ],
            stdout=open(self.root / "serve.log", "ab"),
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                sys.exit(f"the server exited; see {self.root / 'serve.log'}")
            token = sessions / "web-token"
            if token.exists():
                client = Client(f"http://127.0.0.1:{self.port}", token_file=token)
                if client.health():
                    return client
            time.sleep(0.2)
        sys.exit("the server did not come up")

    def kill9(self) -> None:
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


def committed_text(session) -> str:
    return "".join(m["content"] for m in session.messages() if m["role"] == "assistant")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--superfluid", default=shutil.which("superfluid") or "superfluid")
    ap.add_argument("--after", type=int, default=80, help="kill after this many streamed pieces of text")
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="superfluid-plug-"))
    server = Server(args.superfluid, args.model, root)
    try:
        sf = server.start()
        story = sf.session(Sampling(temperature=0.8, seed=2026), system="You are a storyteller. Answer directly, without deliberating.")
        story.user("Tell a story, about 300 words, about a lighthouse keeper who receives a letter from the sea.")

        print(f"{BOLD}session {story.id}{RESET}, streaming...\n")
        shown, pieces = "", 0
        for chunk in story.stream(args.max_tokens):
            if chunk.kind != "text":
                continue
            print(chunk.text, end="", flush=True)
            shown += chunk.text
            pieces += 1
            if pieces == args.after:
                server.kill9()
                break

        print(f"\n\n{RED}{BOLD}  ✗ kill -9 {server.proc.pid}{RESET}\n")
        t0 = time.time()
        sf = server.start()
        print(f"{GREEN}  ✓ restarted in {time.time() - t0:.1f}s{RESET}, replayed the session log\n")

        story = sf.get(story.id)
        kept = committed_text(story)
        lost = len(shown) - len(kept)
        print(f"{DIM}{kept}{RESET}", end="", flush=True)
        reply = story.generate(args.max_tokens, on_text=lambda c: c.kind == "text" and print(c.text, end="", flush=True))
        ahead = f", {lost} streamed ahead of their commit and were drawn again" if lost > 0 else ""
        print(f"\n\n{BOLD}resumed mid-sentence{RESET}: {len(kept)} characters were durable before the kill{ahead}; "
              f"{reply.tokens} tokens after it, finish={reply.finish}.")
    finally:
        server.stop()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
