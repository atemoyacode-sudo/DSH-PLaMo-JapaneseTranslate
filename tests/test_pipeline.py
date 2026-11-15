"""The harness end to end against the stub model, with a fake translator (no GPU, no model files).

  python -m unittest discover -s tests -t .      (from the repository root)
"""

import json
import os
import re
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from harness import stub_model
from harness.pipeline import Harness
from harness.proxy import make_handler
from harness.translator import ChatTranslator, PlamoTranslator, is_japanese, make_translator, pfml3_prompt

PLAMO_DIR = Path(os.environ.get("PLAMO_DIR", Path(__file__).resolve().parents[1] / "models/plamo-3-translate-0.8b-MLX-4bit"))
CODE = "```python\ndef sum_squares(n):\n    # add up the squares\n    return sum(k * k for k in range(1, n + 1))\n```"


class FakeTranslator(PlamoTranslator):
    """Reads the PFML3 prompt like PLaMo would and 'translates' by tagging the text."""

    def _generate(self, prompt, max_tokens):
        text = re.findall(r"<\|plamo:tag\|>input(?:<\|plamo:key\|>lang<\|plamo:val\|>\w+)?<\|plamo:msg\|>(.*?)<\|plamo:tag\|>output",
                          prompt, re.S)[-1]
        target = re.search(r"<\|plamo:tag\|>output<\|plamo:key\|>lang<\|plamo:val\|>([a-z-]+)(?:<\|plamo:key\|>form<\|plamo:val\|>\w+)?<\|plamo:msg\|>$",
                           prompt).group(1)
        return ("【訳】" + text) if target == "ja" else ("EN(" + text + ")")


def post(url, body):
    """The reply's body: parsed JSON, or the raw lines of a stream."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().splitlines() if body.get("stream") else json.load(r)


class PromptTests(unittest.TestCase):
    def test_matches_the_models_chat_template(self):
        try:
            import jinja2
        except ImportError:
            self.skipTest("jinja2 not installed")
        tpl_path = PLAMO_DIR / "chat_template.jinja"
        if not tpl_path.exists():
            self.skipTest(f"no {tpl_path} (set PLAMO_DIR)")

        def boom(msg):
            raise ValueError(msg)

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = boom
        tpl = env.from_string(tpl_path.read_text())
        cases = [
            dict(text="会議は来週の月曜日に変更されました。", target_lang="en"),
            dict(text="This system is convenient.", target_lang="ja", form="polite", source_lang="en"),
            dict(text="かみを切りました。", target_lang="en", context="工作で折り紙を作っています。"),
            dict(text="オーナーでフィルタリングすることもできます。", target_lang="en", source_lang="ja",
                 history=[("プロジェクトの管理画面を開きます。", "Open the project dashboard.")],
                 glossary=[("管理画面", "dashboard")]),
            dict(text="  spaces\n kept \n", target_lang="ja"),
        ]
        for c in cases:
            msgs = []
            for s, t in c.get("history", []):
                msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": t}]
            msgs.append({"role": "user", "content": c["text"]})
            kw = {k: c[k] for k in ("source_lang", "context", "form") if k in c}
            if "glossary" in c:
                kw["glossary"] = [{"source": s, "target": t} for s, t in c["glossary"]]
            want = tpl.render(messages=msgs, target_lang=c["target_lang"], add_generation_prompt=True,
                              bos_token="<|plamo:bos|>", **kw)
            self.assertEqual("<|plamo:bos|>" + pfml3_prompt(**c), want, c)

    def test_rejects_what_the_template_rejects(self):
        for bad in (dict(text="x <|plamo:tag|>", target_lang="en"), dict(text=" ", target_lang="en"),
                    dict(text="x", target_lang="EN"), dict(text="x", target_lang="en", form="polite")):
            with self.assertRaises(ValueError):
                pfml3_prompt(**bad)

    def test_is_japanese(self):
        for t in ("こんにちは", "素数とは何ですか?", "東京", "`code` を説明して"):
            self.assertTrue(is_japanese(t), t)
        for t in ("hello", "你好，请问今天天气怎么样？我想去公园散步。", "`こんにちは`", "x = 1", ""):
            self.assertFalse(is_japanese(t), t)


class ChatStub:
    """A chat model that 'translates' by tagging, thinks out loud, and can reject chat_template_kwargs like a
    hosted API."""

    def __init__(self, reject_kwargs=False):
        from http.server import BaseHTTPRequestHandler
        stub = self
        self.requests = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append((body, self.headers.get("Authorization")))
                if reject_kwargs and "chat_template_kwargs" in body:
                    data, code = b'{"error": "Unrecognized request argument: chat_template_kwargs"}', 400
                else:
                    text = "<think>hmm</think>\n訳:" + body["messages"][-1]["content"]
                    data, code = json.dumps({"choices": [{"message": {"content": text}}]}).encode(), 200
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


class ChatTranslatorTests(unittest.TestCase):
    def test_translates_with_an_instruction(self):
        stub = ChatStub()
        try:
            tr = make_translator(stub.url, "chat", model="qwen", api_key="k", glossary=[("管理画面", "dashboard")])
            self.assertIsInstance(tr, ChatTranslator)
            self.assertEqual(tr.credit, "")
            out = tr.translate("Open the dashboard.", "en", "ja", history=[("Hi.", "やあ。")])
            self.assertEqual(out, "訳:Open the dashboard.")  # thinking stripped
            body, auth = stub.requests[-1]
            self.assertEqual(auth, "Bearer k")
            self.assertEqual(body["model"], "qwen")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            system = body["messages"][0]["content"]
            self.assertIn("from English into Japanese", system)
            self.assertIn("です・ます", system)
            self.assertIn("dashboard -> 管理画面", system)  # the glossary reversed for en -> ja
            self.assertEqual([m["role"] for m in body["messages"]], ["system", "user", "assistant", "user"])
        finally:
            stub.close()

    def test_falls_back_when_the_server_rejects_template_kwargs(self):
        stub = ChatStub(reject_kwargs=True)
        try:
            tr = make_translator(stub.url, "chat")
            self.assertEqual(tr.translate("こんにちは", "ja", "en"), "訳:こんにちは")
            self.assertEqual(tr.translate("さようなら", "ja", "en"), "訳:さようなら")
            self.assertEqual(len(stub.requests), 3)  # one rejected, then never sent again
            self.assertNotIn("chat_template_kwargs", stub.requests[-1][0])
        finally:
            stub.close()

    def test_plamo_kinds_and_credit(self):
        self.assertEqual(make_translator("http://127.0.0.1:1/v1").credit, "Built with PLaMo")
        with self.assertRaises(ValueError):
            make_translator("/some/dir", "chat")
        with self.assertRaises(ValueError):
            make_translator("http://x/v1", "deepl")


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub, cls.state = stub_model.serve(0)
        cls.stub2, cls.state2 = stub_model.serve(0, split_reasoning=True)

    @classmethod
    def tearDownClass(cls):
        cls.stub.shutdown(); cls.stub.server_close()
        cls.stub2.shutdown(); cls.stub2.server_close()

    def harness(self, stub):
        return Harness(FakeTranslator(), base_url=f"http://127.0.0.1:{stub.server_port}/v1")

    def run_one(self, stub, question):
        h = self.harness(stub)
        msgs, target = h.prepare([{"role": "user", "content": question}])
        events = list(h.run(msgs, target))
        return msgs, target, events

    def test_japanese_round_trip(self):
        for stub, state in ((self.stub, self.state), (self.stub2, self.state2)):
            msgs, target, events = self.run_one(stub, "二乗の和を求める関数を書いて")
            self.assertEqual(target, "ja")
            self.assertEqual(state["last"]["messages"][0]["content"], "EN(二乗の和を求める関数を書いて)")
            answer = "".join(e.text for e in events if e.kind == "text")
            fin = events[-1].data
            self.assertEqual(answer, fin["answer"])
            self.assertIn(CODE, answer)
            self.assertIn("$$\n\\frac{n(n+1)(2n+1)}{6}\n$$", answer)
            self.assertIn("`sum_squares(n)`", answer)
            self.assertIn("$\\sum_{k=1}^{n} k^2$", answer)
            self.assertIn("1. 【訳】Loop over the numbers from 1 to `n`.", answer)
            self.assertIn("| 【訳】Input | 【訳】Output |\n|---|---|\n| 3 | 14 |", answer)
            self.assertIn("https://docs.python.org/3/", answer)
            self.assertNotIn("I will answer", answer)  # thinking is not in the answer
            self.assertIn("I will answer", "".join(e.text for e in events if e.kind == "reasoning"))
            self.assertTrue(fin["english"].startswith("You asked: EN("))

    def test_english_is_left_alone(self):
        msgs, target, events = self.run_one(self.stub, "Write a sum of squares function")
        self.assertIsNone(target)
        self.assertEqual(msgs[0]["content"], "Write a sum of squares function")
        answer = "".join(e.text for e in events if e.kind == "text")
        self.assertNotIn("【訳】", answer)
        self.assertTrue(answer.startswith("You asked: Write a sum"))

    def test_thinking_off(self):
        h = self.harness(self.stub)
        msgs, target = h.prepare([{"role": "user", "content": "こんにちは"}])
        events = list(h.run(msgs, target, {"chat_template_kwargs": {"enable_thinking": False}}))
        self.assertFalse(self.state["last"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(self.state["last"]["temperature"], 0.7)
        self.assertFalse(any(e.kind == "reasoning" for e in events))


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub, cls.state = stub_model.serve(0)
        cls.h = Harness(FakeTranslator(), base_url=f"http://127.0.0.1:{cls.stub.server_port}/v1")
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.h, "astrantia-ja", False))
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.proxy.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown(); cls.proxy.server_close()
        cls.stub.shutdown(); cls.stub.server_close()

    def test_models(self):
        with urllib.request.urlopen(self.url + "/models") as r:
            j = json.load(r)
        self.assertEqual(j["data"][0]["id"], "astrantia-ja")
        self.assertEqual(j["data"][0]["description"], "Built with PLaMo")

    def test_non_stream_and_stream_agree(self):
        body = {"model": "astrantia-ja", "messages": [{"role": "user", "content": "二乗の和は?"}]}
        j = (post(self.url + "/chat/completions", body))
        msg = j["choices"][0]["message"]
        self.assertIn(CODE, msg["content"])
        self.assertNotIn("reasoning_content", msg)
        self.assertEqual(j["choices"][0]["finish_reason"], "stop")
        lines = post(self.url + "/chat/completions", {**body, "stream": True, "stream_options": {"include_usage": True}})
        parts, usage, done = [], None, False
        for line in lines:
            line = line.strip()
            if line == b"data: [DONE]":
                done = True
            elif line.startswith(b"data: "):
                c = json.loads(line[6:])
                usage = c.get("usage") or usage
                for ch in c["choices"]:
                    parts.append(ch["delta"].get("content") or "")
        self.assertTrue(done)
        self.assertEqual("".join(parts), msg["content"])
        self.assertIsNotNone(usage)

    def test_history_uses_the_models_own_english(self):
        body = {"messages": [{"role": "user", "content": "二乗の和は?"}]}
        ja = (post(self.url + "/chat/completions", body))["choices"][0]["message"]["content"]
        body["messages"] += [{"role": "assistant", "content": ja}, {"role": "user", "content": "ありがとう"}]
        post(self.url + "/chat/completions", body)
        sent = self.state["last"]["messages"]
        self.assertTrue(sent[1]["content"].startswith("You asked: EN(二乗の和は?)"))
        self.assertNotIn("【訳】", sent[1]["content"])
        self.assertEqual(sent[2]["content"], "EN(ありがとう)")

    def test_tool_calls_pass_through(self):
        tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
        body = {"messages": [{"role": "user", "content": "東京の天気を調べて"}], "tools": tools}
        j = (post(self.url + "/chat/completions", body))
        self.assertEqual(self.state["last"]["tools"], tools)
        c = j["choices"][0]
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual(c["message"]["tool_calls"][0]["function"],
                         {"name": "search", "arguments": '{"query": "東京の天気"}'})
        body["messages"] += [c["message"], {"role": "tool", "tool_call_id": "call_1", "content": "晴れ、18度"}]
        post(self.url + "/chat/completions", {**body, "tools": []})
        sent = self.state["last"]["messages"]
        self.assertEqual(sent[1]["tool_calls"], c["message"]["tool_calls"])
        self.assertEqual(sent[2]["content"], "晴れ、18度")  # tool results are not translated

    def test_unreachable_model_is_502(self):
        h = Harness(FakeTranslator(), base_url="http://127.0.0.1:9/v1")
        p = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(h, "x", False))
        threading.Thread(target=p.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post(f"http://127.0.0.1:{p.server_port}/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "やあ"}], "stream": True})
            self.assertEqual(cm.exception.code, 502)
        finally:
            p.shutdown(); p.server_close()


if __name__ == "__main__":
    unittest.main()
