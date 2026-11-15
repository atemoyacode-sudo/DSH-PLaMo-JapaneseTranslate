"""A tiny stand-in for the model's OpenAI-compatible server, for tests and for trying the harness without a GPU.

  python -m harness.stub_model --port 8093 [--split-reasoning]

It streams (or returns) a fixed English answer that has prose, a list, inline code, inline and display math, a code
block and a table, after a short "thinking" part: inside `content` ending in </think> (what Qwen3.5 under a plain
server produces) or, with --split-reasoning, in `reasoning_content` (what vLLM's reasoning parser produces). If the
request offers tools, it answers with a tool call instead. GET /last returns the last request body it received.
It never runs anything it receives.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THINKING = "The user asks a question. I will answer with a short example of Python and a formula."
ANSWER = """You asked: {question}

Here is a short answer. The function `sum_squares(n)` returns $\\sum_{{k=1}}^{{n}} k^2$, which equals:

$$
\\frac{{n(n+1)(2n+1)}}{{6}}
$$

Steps:
1. Loop over the numbers from 1 to `n`.
2. Add each square to a running total.

```python
def sum_squares(n):
    # add up the squares
    return sum(k * k for k in range(1, n + 1))
```

| Input | Output |
|---|---|
| 3 | 14 |

See https://docs.python.org/3/ for more."""


def answer_for(body: dict) -> str:
    q = ""
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            q = m["content"] if isinstance(m["content"], str) else " ".join(p.get("text", "") for p in m["content"])
            break
    return ANSWER.format(question=q.strip())


def make_handler(split_reasoning: bool, state: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/last":
                self._json(state.get("last") or {})
            else:
                self._json({"object": "list", "data": [{"id": "default_model", "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            state["last"] = body
            thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
            if body.get("tools"):
                name = body["tools"][0]["function"]["name"]
                deltas = [{"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                           "function": {"name": name, "arguments": ""}}]},
                          {"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "東京の天気"}'}}]}]
                finish = "tool_calls"
            else:
                text = answer_for(body)
                deltas = []
                if thinking and split_reasoning:
                    deltas += [{"reasoning_content": w} for w in _words(THINKING)]
                elif thinking:
                    text = THINKING + "\n</think>\n\n" + text
                deltas += [{"content": w} for w in _words(text)]
                finish = "stop"
            usage = {"prompt_tokens": 10, "completion_tokens": len(deltas), "total_tokens": 10 + len(deltas)}
            if not body.get("stream"):
                msg = {"role": "assistant", "content": "".join(d.get("content", "") for d in deltas) or None}
                if body.get("tools"):
                    msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
                        "name": deltas[0]["tool_calls"][0]["function"]["name"], "arguments": '{"query": "東京の天気"}'}}]
                if split_reasoning and thinking:
                    msg["reasoning_content"] = "".join(d.get("reasoning_content", "") for d in deltas)
                self._json({"choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for d in deltas:
                self._chunk({"choices": [{"index": 0, "delta": d, "finish_reason": None}]})
            self.wfile.write(b": ping\n\n")
            self._chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            self._chunk({"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _chunk(self, obj):
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def _words(text: str) -> list[str]:
    """Split into small pieces the way a tokenizer would stream them (spaces and newlines kept)."""
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in " \n" and len(cur) > 2:
            out.append(cur)
            cur = ""
    return out + ([cur] if cur else [])


def serve(port: int = 0, split_reasoning: bool = False) -> "tuple[ThreadingHTTPServer, dict]":
    """Start in a background thread (for tests); returns the server and its state ({"last": request body})."""
    state: dict = {}
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(split_reasoning, state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.stub_model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--split-reasoning", action="store_true")
    args = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.split_reasoning, {}))
    print(f"stub model on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
