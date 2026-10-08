"""Offline tests: a fake `--web` server checks the wire shapes the client sends and parses.

Run with `python -m unittest discover clients/python/tests`. Set SUPERFLUID_WEB_URL (and the
token as for `Client()`) to also run `LiveTest` against a real server.
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from superfluid_client import Client, Sampling, SessionBusy, SuperfluidError  # noqa: E402

TOKEN = "t0k"


def event(eid, kind, **data):
    return {"event_id": eid, "epoch": 1, "ts_unix_ms": 0, "body": {kind: data} if data else kind}


class Fake:
    """Records requests and answers from a script keyed by verb."""

    def __init__(self):
        self.requests = []
        self.stream_bodies = []
        self.frames = []
        self.answers = {}


def handler(fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _authorized(self):
            ok = (
                self.headers.get("authorization") == f"Bearer {TOKEN}"
                and self.headers.get("x-superfluid-csrf") == TOKEN
            )
            if not ok:
                self._json(401, {"Err": {"message": "bad or missing bearer token"}})
            return ok

        def _json(self, status, obj):
            raw = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.send_response(200 if self.path == "/web/health" else 404)
            self.send_header("content-length", "0")
            self.end_headers()

        def do_POST(self):
            if not self._authorized():
                return
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/web/stream":
                fake.stream_bodies.append(body)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for f in fake.frames:
                    self.wfile.write(f"data: {json.dumps(f)}\n\n".encode())
                self.wfile.write(b": keep-alive\n\ndata: [DONE]\n\n")
                return
            fake.requests.append(body)
            ((verb, _),) = body.items()
            answer = fake.answers.get(verb)
            self._json(200, answer(body[verb]) if callable(answer) else answer)

    return H


class FakeServerTest(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.fake))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}", token=TOKEN)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_create_sends_full_params_and_system_message(self):
        self.fake.answers = {
            "Create": {"Created": {"session": 7}},
            "AppendMessage": {"Committed": {"event": event(1, "Message", role=0, text="hi", span=[1])}},
        }
        s = self.client.session(Sampling(temperature=0.5, top_p=0.9, min_p=0.05, top_k=10, seed=3), system="hi")
        self.assertEqual(s.id, 7)
        self.assertEqual(
            self.fake.requests[0],
            {"Create": {"parent": None, "params": {"temperature": 0.5, "top_p": 0.9, "min_p": 0.05, "top_k": 10, "seed": 3}}},
        )
        self.assertEqual(self.fake.requests[1], {"AppendMessage": {"session": 7, "role": 0, "text": "hi"}})

    def test_session_with_tools_declares_them_and_answers_a_call(self):
        weather = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        bare = '{"name": "now"}'
        self.fake.answers = {
            "Create": {"Created": {"session": 3}},
            "AppendSystem": {"Committed": {"event": event(1, "Message", role=0, text="sys", span=[1])}},
            "AppendToolResult": {"Committed": {"event": event(5, "ToolResult", call_id=4, content="18C", span=[2])}},
        }
        self.fake.frames = [
            {"committed": event(4, "ToolUse", name="get_weather", arguments='{"city": "Paris"}')},
            {"done": {"tokens_generated": 9, "finish": 1, "warm_prefix": 0}},
        ]
        s = self.client.session(Sampling.greedy(), system="sys", tools=[weather, bare])
        self.assertEqual(
            self.fake.requests[1],
            {"AppendSystem": {"session": 3, "text": "sys", "tools": [json.dumps(weather), bare]}},
        )
        self.assertEqual(len(self.fake.requests), 2, "the system text goes with the tools, not as a second message")
        (call,) = s.generate(64).tool_calls
        self.assertEqual((call.id, call.name, call.json()), (4, "get_weather", {"city": "Paris"}))
        s.tool_result(call, "18C")
        self.assertEqual(self.fake.requests[-1], {"AppendToolResult": {"session": 3, "call_id": 4, "content": "18C"}})

    def test_declare_refusal_raises(self):
        from superfluid_client import Session

        self.fake.answers = {"AppendSystem": {"Err": {"message": "tool 0 is not a function declaration: it has no name"}}}
        with self.assertRaisesRegex(SuperfluidError, "tool 0 is not a function declaration"):
            Session(self.client, 2, Sampling.greedy()).declare(tools=[{"type": "function", "function": {}}])
        self.assertEqual(self.fake.requests[-1]["AppendSystem"]["text"], None)

    def test_fork_defaults_to_head_and_reseeds(self):
        self.fake.answers = {
            "Read": {"Events": {"events": [event(0, "Created", parent=None, params={}), event(1, "EpochBump")]}},
            "Fork": {"Created": {"session": 9}},
        }
        from superfluid_client import Session

        parent = Session(self.client, 4, Sampling(temperature=1.0, seed=1))
        child = parent.fork(seed=2)
        fork = self.fake.requests[-1]["Fork"]
        self.assertEqual((fork["session"], fork["at_event"]), (4, 2))
        self.assertEqual(fork["params"]["seed"], 2)
        self.assertEqual(fork["params"]["temperature"], 1.0)
        self.assertEqual((child.id, child.sampling.seed), (9, 2))
        parent.fork_after(5)
        self.assertEqual(self.fake.requests[-1]["Fork"], {"session": 4, "at_event": 6, "params": None})

    def test_stream_yields_provisional_and_builds_reply(self):
        self.fake.answers = {"Create": {"Created": {"session": 1}}}
        self.fake.frames = [
            {"provisional": {"channel": 1, "text": "hm"}},
            {"provisional": {"channel": 0, "text": "Hel"}},
            {"provisional": {"channel": 0, "text": "lo"}},
            {"committed": event(5, "Generated", span=[1], text="hm", channel=1, finish=0)},
            {"committed": event(6, "Generated", span=[2, 3], text="Hello", channel=0, finish=0)},
            {"committed": event(7, "ToolUse", name="get", arguments='{"q": 1}')},
            {"done": {"tokens_generated": 3, "finish": 1, "warm_prefix": 12}},
        ]
        s = self.client.session(Sampling.greedy())
        st = s.stream(32)
        chunks = [(c.kind, c.text) for c in st]
        self.assertEqual(chunks, [("reasoning", "hm"), ("text", "Hel"), ("text", "lo")])
        self.assertEqual(self.fake.stream_bodies[-1], {"session": 1, "max_tokens": 32, "provisional": True})
        r = st.reply
        self.assertEqual((r.text, r.reasoning, r.finish, r.tokens, r.warm_prefix), ("Hello", "hm", "stop", 3, 12))
        self.assertEqual(r.tool_calls[0].id, 7)
        self.assertEqual(r.tool_calls[0].json(), {"q": 1})

    def test_generate_without_callback_uses_committed_text(self):
        self.fake.answers = {"Create": {"Created": {"session": 1}}}
        self.fake.frames = [
            {"committed": event(4, "Generated", span=[1], text="ok", channel=0, finish=2)},
            {"done": {"tokens_generated": 1, "finish": 2, "warm_prefix": 0}},
        ]
        r = self.client.session(Sampling.greedy()).generate(1)
        self.assertEqual((r.text, r.finish), ("ok", "length"))
        self.assertFalse(self.fake.stream_bodies[-1]["provisional"])

    def test_stream_error_frame_raises(self):
        self.fake.answers = {"Create": {"Created": {"session": 1}}}
        self.fake.frames = [{"error": "session 1 is generating; append/generate after it finishes or cancel it"}]
        with self.assertRaises(SessionBusy):
            self.client.session(Sampling.greedy()).generate(4)
        self.assertNotIn("Cancel", [next(iter(r)) for r in self.fake.requests], "another client's generation is left alone")

    def test_leaving_a_stream_early_cancels_and_drains(self):
        self.fake.answers = {"Create": {"Created": {"session": 1}}, "Cancel": "Cancelling"}
        self.fake.frames = [
            {"provisional": {"channel": 0, "text": "a"}},
            {"provisional": {"channel": 0, "text": "b"}},
            {"done": {"tokens_generated": 2, "finish": 4, "warm_prefix": 0}},
        ]
        st = self.client.session(Sampling.greedy()).stream(100)
        for _ in st:
            break
        self.assertEqual(self.fake.requests[-1], {"Cancel": {"session": 1}})

    def test_refusals_and_auth(self):
        self.fake.answers = {"Read": {"Err": {"message": "unknown session 3"}}}
        from superfluid_client import Session

        with self.assertRaisesRegex(SuperfluidError, "unknown session 3"):
            Session(self.client, 3, Sampling.greedy()).events()
        bad = Client(self.client.url, token="nope")
        with self.assertRaises(SuperfluidError) as cm:
            bad.tree()
        self.assertEqual(cm.exception.status, 401)

    def test_cancel_reports_whether_anything_was_running(self):
        from superfluid_client import Session

        s = Session(self.client, 2, Sampling.greedy())
        self.fake.answers = {"Cancel": "Cancelling"}
        self.assertTrue(s.cancel())
        self.fake.answers = {"Cancel": {"Err": {"message": "session 2 has no generation in flight"}}}
        self.assertFalse(s.cancel())

    def test_edit_replaces_from_the_message_to_the_head(self):
        msg = event(2, "Message", role=1, text="old", span=[1])
        self.fake.answers = {
            "Read": lambda a: {"Events": {"events": [e for e in [event(0, "Created", parent=None, params={}), msg, event(3, "EpochBump")] if e["event_id"] >= a["cursor"]]}},
            "Rebase": {"Created": {"session": 5}},
        }
        from superfluid_client import Session

        child = Session(self.client, 1, Sampling.greedy()).edit(2, "new")
        self.assertEqual(child.id, 5)
        self.assertEqual(
            self.fake.requests[-1]["Rebase"]["edits"],
            [{"Replace": {"from": 2, "to": 4, "role": 1, "text": "new"}}],
        )

    def test_messages_fold_generated_events(self):
        self.fake.answers = {
            "Read": {
                "Events": {
                    "events": [
                        event(0, "Created", parent=None, params={}),
                        event(1, "Message", role=1, text="q", span=[]),
                        event(2, "GenerationPrompt", span=[]),
                        event(3, "Generated", span=[], text="think", channel=1, finish=0),
                        event(4, "Generated", span=[], text="a", channel=0, finish=0),
                        event(5, "Generated", span=[], text="b", channel=0, finish=1),
                        event(6, "Message", role=1, text="q2", span=[]),
                    ]
                }
            }
        }
        from superfluid_client import Session

        msgs = Session(self.client, 1, Sampling.greedy()).messages()
        self.assertEqual(
            [(m["role"], m["content"], m.get("reasoning")) for m in msgs],
            [("user", "q", None), ("assistant", "ab", "think"), ("user", "q2", None)],
        )


@unittest.skipUnless(os.environ.get("SUPERFLUID_WEB_URL"), "set SUPERFLUID_WEB_URL to test a live server")
class LiveTest(unittest.TestCase):
    def test_fork_is_warm_and_cancel_then_continue_is_seamless(self):
        sf = Client()
        base = sf.session(Sampling(temperature=0.9, top_k=40, seed=11), system="You are a storyteller.")
        q = base.user("Tell a short story about a lighthouse.")
        whole = base.fork().generate(160)
        cut = base.fork()
        st = cut.stream(160)
        for i, _ in enumerate(st):
            if i == 20:
                break
        rest = cut.generate(160)
        text = "".join(m.get("reasoning", "") + m["content"] for m in cut.messages() if m["role"] == "assistant")
        self.assertTrue(text.startswith(whole.reasoning + whole.text), (text, whole))
        self.assertGreater(rest.warm_prefix, 0)
        branch = base.fork_after(q, seed=12)
        self.assertGreater(branch.generate(8).warm_prefix, 0)


if __name__ == "__main__":
    unittest.main()
