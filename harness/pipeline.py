"""Japanese in, English to the model, Japanese out.

`Harness.prepare` turns the client's messages into English ones (and says which language to answer in);
`Harness.run` sends them to the model's OpenAI-compatible endpoint (always streamed) and yields events: thinking
text (untranslated), answer text (translated a block at a time, as each block completes), tool calls (unchanged)
and the finish. The CLI prints the events; the proxy turns them into OpenAI responses.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

from .protect import BlockStream, Translate, translate_block, translate_markdown
from .translator import Translator, is_japanese

# Qwen3.5's recommended sampling, used when the client sends none.
SAMPLING = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20},  # thinking on
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}  # thinking off
# Fields passed through from a client's request to the model.
PASS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "stop", "seed", "tools",
        "tool_choice", "parallel_tool_calls", "presence_penalty", "frequency_penalty", "repetition_penalty",
        "response_format")
HISTORY_CHARS = 600  # previous paragraph pair given to PLaMo as history, if both sides are at most this long


@dataclass
class Event:
    kind: str  # "reasoning" | "text" | "tool_calls" | "finish" | "keepalive"
    text: str = ""
    data: Any = None


class UpstreamError(RuntimeError):
    pass


@dataclass
class Harness:
    translator: Translator
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "default_model"
    api_key: "str | None" = None
    thinking: bool = True
    max_tokens: int = 8192
    timeout: float = 1800.0
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Japanese text we sent to a client -> the English the model wrote (a chat app sends our Japanese back as
        # history; the model should see its own words, not a translation of a translation).
        self._english: OrderedDict = OrderedDict()

    # ---- translation -------------------------------------------------------------------------------------------
    def _translate_fn(self, source: str, target: str) -> Translate:
        """A translate function for one message: it gives PLaMo the previous paragraph pair as history, which keeps
        terms and style consistent across the paragraphs of one answer."""
        last: list = []

        def tr(text: str, context: "str | None") -> str:
            hist = last[-1:] if last and len(last[-1][0]) <= HISTORY_CHARS and len(last[-1][1]) <= HISTORY_CHARS else None
            t0 = time.monotonic()
            out = self.translator.translate(text, source, target, context=context, history=hist)
            self.stats["translate_s"] = self.stats.get("translate_s", 0.0) + time.monotonic() - t0
            self.stats["translated_chars"] = self.stats.get("translated_chars", 0) + len(text)
            if context is None:
                last.append((text, out))
            return out

        return tr

    def to_english(self, text: str) -> str:
        key = text.strip()
        if key in self._english:
            return self._english[key]
        return translate_markdown(text, self._translate_fn("ja", "en"), skip=lambda b: not is_japanese(b))

    def remember(self, japanese: str, english: str) -> None:
        self._english[japanese.strip()] = english
        while len(self._english) > 512:
            self._english.popitem(last=False)

    # ---- messages ----------------------------------------------------------------------------------------------
    def prepare(self, messages: list[dict]) -> "tuple[list[dict], str | None]":
        """English messages for the model, and "ja" if the answer should be translated to Japanese (the last user
        message is Japanese) else None. Tool calls and tool results pass unchanged."""
        target = None
        for m in reversed(messages):
            if m.get("role") == "user":
                target = "ja" if is_japanese(_text_of(m.get("content"))) else None
                break
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") in ("user", "system", "developer", "assistant") and m.get("content"):
                m["content"] = self._content_to_english(m["content"])
            out.append(m)
        return out, target

    def _content_to_english(self, content: Any) -> Any:
        if isinstance(content, str):
            return self.to_english(content) if is_japanese(content) else content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text" and is_japanese(p.get("text", "")):
                    p = {**p, "text": self.to_english(p["text"])}
                parts.append(p)
            return parts
        return content

    # ---- the model ---------------------------------------------------------------------------------------------
    def body(self, messages: list[dict], client: "dict | None" = None) -> dict:
        client = client or {}
        kwargs = dict(client.get("chat_template_kwargs") or {})
        kwargs.setdefault("enable_thinking", self.thinking)
        body = {k: client[k] for k in PASS if k in client}
        for k, v in SAMPLING[bool(kwargs["enable_thinking"])].items():
            body.setdefault(k, v)
        body.setdefault("max_tokens", self.max_tokens)
        body.update(model=self.model, messages=messages, stream=True, chat_template_kwargs=kwargs,
                    stream_options={"include_usage": True})
        return body

    def _open(self, body: dict):
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise UpstreamError(f"model server answered {e.code}: {e.read()[:500].decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise UpstreamError(f"cannot reach the model server at {self.base_url}: {e.reason}") from e

    def run(self, messages_en: list[dict], target: "str | None", client: "dict | None" = None) -> Iterator[Event]:
        """Stream the model's answer. Answer text is translated to `target` a block at a time (None: as is).
        The final "finish" event carries {"finish_reason", "usage", "english", "answer"}.
        Connects before returning, so that an unreachable or failing model raises UpstreamError here."""
        body = self.body(messages_en, client)
        t0 = time.monotonic()
        return self._events(self._open(body), body, target, t0)

    def _events(self, resp, body: dict, target: "str | None", t0: float) -> Iterator[Event]:
        self.stats.update(translate_s=0.0, translated_chars=0)
        tr = self._translate_fn("en", "ja") if target == "ja" else None
        split = ThinkSplit(bool(body["chat_template_kwargs"].get("enable_thinking")))
        blocks = BlockStream()
        english, answer, calls = [], [], {}
        finish, usage, first = None, None, True

        def emit(new_blocks):
            for b in new_blocks:
                english.append(b.text)
                out = translate_block(b, tr, skip=is_japanese) if tr else b.text
                answer.append(out)
                yield Event("text", out)

        def answer_text(a: str):
            nonlocal first
            if first:
                a = a.lstrip()
                first = not a
            if a:
                yield from emit(blocks.feed(a))

        with resp:
            for chunk in _sse(resp):
                if chunk is None:
                    yield Event("keepalive")
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    r = d.get("reasoning_content") or d.get("reasoning")
                    if r:
                        split.server_splits()
                        yield Event("reasoning", r)
                    if d.get("content"):
                        r2, a = split.feed(d["content"])
                        if r2:
                            yield Event("reasoning", r2)
                        if a:
                            yield from answer_text(a)
                        elif not r2:
                            yield Event("keepalive")  # thinking still being collected
                    for tc in d.get("tool_calls") or []:
                        _merge_call(calls, tc)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        r2, a = split.finish()
        if r2:
            yield Event("reasoning", r2)
        yield from answer_text(a)
        yield from emit(blocks.flush())
        if calls:
            yield Event("tool_calls", data=[calls[k] for k in sorted(calls)])
        en, ja = "".join(english), "".join(answer)
        if tr and ja.strip():
            self.remember(ja, en)
        self.stats["total_s"] = time.monotonic() - t0
        yield Event("finish", data={"finish_reason": finish or "stop", "usage": usage, "english": en, "answer": ja})


class ThinkSplit:
    """Separates thinking from the answer when the server leaves both in `content` (Qwen3.5's template opens
    <think> in the prompt, so the text is "thinking</think>answer"). Once the server is seen to send thinking in a
    field of its own, content is all answer."""

    def __init__(self, thinking: bool) -> None:
        self.waiting = thinking  # content may still be thinking
        self.buf = ""

    def server_splits(self) -> None:
        self.waiting = False

    def feed(self, s: str) -> "tuple[str, str]":
        if not self.waiting:
            out, self.buf = self.buf + s, ""
            return "", out
        self.buf += s
        if "</think>" in self.buf:
            r, a = self.buf.split("</think>", 1)
            self.waiting, self.buf = False, ""
            return _no_open(r), a
        return "", ""

    def finish(self) -> "tuple[str, str]":
        text, self.buf = self.buf, ""
        if not self.waiting:
            return "", text
        self.waiting = False
        if text.lstrip().startswith("<think>"):  # thinking that never closed: there is no answer
            return _no_open(text), ""
        return "", text  # no </think> at all: the server stripped the tags, or the model did not think


def _no_open(r: str) -> str:
    r = r.lstrip()
    return r[len("<think>"):] if r.startswith("<think>") else r


def _merge_call(calls: dict, tc: dict) -> None:
    i = tc.get("index", len(calls))
    c = calls.setdefault(i, {"index": i, "id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    if tc.get("id"):
        c["id"] = tc["id"]
    f = tc.get("function") or {}
    if f.get("name"):
        c["function"]["name"] += f["name"]
    if f.get("arguments"):
        c["function"]["arguments"] += f["arguments"] if isinstance(f["arguments"], str) else json.dumps(f["arguments"])


def _sse(resp) -> Iterator["dict | None"]:
    """Chunks of an OpenAI stream; None for a comment line (a keep-alive). A non-stream JSON reply is accepted too
    and turned into one chunk."""
    first = resp.readline()
    if first.lstrip().startswith(b"{"):  # the server ignored stream=true
        j = json.loads(first + resp.read())
        msg = j["choices"][0].get("message", {})
        yield {"choices": [{"delta": msg, "finish_reason": j["choices"][0].get("finish_reason")}],
               "usage": j.get("usage")}
        return
    line = first
    while line:
        s = line.strip()
        if s.startswith(b"data:"):
            data = s[5:].strip()
            if data == b"[DONE]":
                return
            if data:
                yield json.loads(data)
        elif s.startswith(b":"):
            yield None
        line = resp.readline()


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""
