"""Translators behind one small interface: PLaMo 3 Translate (the default), or any chat model.

For PLaMo the prompt is PFML3, the format its chat template produces (see the model card's "PFML3 Prompt Format");
the card recommends building it directly and using the Completions API for applications, which is what
`HttpTranslator` does. `MlxTranslator` runs the MLX-4bit build in this process instead (no second server).
`ChatTranslator` asks any chat model behind an OpenAI-compatible server to translate, with an instruction.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field

STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
LANG = re.compile(r"^[a-z]{2,3}(-[a-z]{4})?$")
KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def is_japanese(text: str) -> bool:
    """Japanese if it has kana, or mostly Han characters with no Latin letters around (a kanji-only line).
    Chinese text has no kana and usually more than a few characters, so it is not taken for Japanese."""
    from .protect import find_spans

    plain, last = [], 0
    for a, b in find_spans(text):
        plain.append(text[last:a])
        last = b
    plain.append(text[last:])
    t = "".join(plain)
    if len(KANA.findall(t)) >= 1:
        return True
    han = len(HAN.findall(t))
    latin = sum(ch.isascii() and ch.isalpha() for ch in t)
    return 0 < han <= 8 and latin == 0


def pfml3_prompt(text: str, target_lang: str, source_lang: "str | None" = None, context: "str | None" = None,
                 form: "str | None" = None, glossary: "list[tuple[str, str]] | None" = None,
                 history: "list[tuple[str, str]] | None" = None) -> str:
    """The prompt PLaMo's chat template builds, without the leading <|plamo:bos|> (servers and tokenizers add it).
    Raises ValueError where the template would raise."""
    for name, value in (("text", text), ("context", context or "")):
        if "<|plamo:" in value:
            raise ValueError(f"{name} must not contain reserved PFML tokens (<|plamo:...)")
    if not text.strip():
        raise ValueError("text must not be empty")
    for lang in (target_lang, source_lang):
        if lang is not None and not LANG.match(lang):
            raise ValueError(f"bad language code {lang!r}: use lowercase codes such as en, ja, zh-hans")
    if form is not None and (form not in ("polite", "plain") or target_lang != "ja"):
        raise ValueError('form must be polite or plain and requires target_lang="ja"')
    src = f"<|plamo:key|>lang<|plamo:val|>{source_lang}" if source_lang else ""
    p = "<|plamo:tag|>dataset<|plamo:msg|>translation"
    if context:
        p += f"<|plamo:tag|>context<|plamo:msg|>{context}"
    for s, t in history or []:
        p += f"<|plamo:tag|>input{src}<|plamo:msg|>{s}"
        p += f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target_lang}<|plamo:msg|>{t}"
    if glossary:
        p += "<|plamo:tag|>glossary" + "".join(f"<|plamo:index|>{s}<|plamo:item|>{t}" for s, t in glossary)
    p += f"<|plamo:tag|>input{src}<|plamo:msg|>{text}"
    p += f"<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target_lang}"
    if form:
        p += f"<|plamo:key|>form<|plamo:val|>{form}"
    return p + "<|plamo:msg|>"


def budget(text: str) -> int:
    """Generation limit for one translation: generous (Japanese -> English grows in characters), capped so a looping
    translation cannot run for long."""
    return min(4096, 64 + 2 * len(text))


def clean(out: str) -> str:
    """Drop any PFML token the runtime let through."""
    return out.split("<|plamo:")[0]


@dataclass
class Translator:
    """Base: caching and the edges of the text; subclasses implement `_run`."""

    form: "str | None" = "polite"  # for English -> Japanese
    glossary: "list[tuple[str, str]]" = field(default_factory=list)  # (Japanese, English) pairs
    cache_size: int = 2048
    credit = ""  # what the translator's license asks to be shown ("Built with PLaMo")

    def __post_init__(self) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.calls = 0  # model calls (cache misses), for the measured example

    def translate(self, text: str, source: str, target: str, context: "str | None" = None,
                  history: "list[tuple[str, str]] | None" = None) -> str:
        key = (text, source, target, context, tuple(history or ()))
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        gl = self.glossary if source == "ja" else [(e, j) for j, e in self.glossary]
        self.calls += 1
        out = self._run(text, source, target, context, history, gl, self.form if target == "ja" else None)
        # Keep the edges exactly so that layout survives.
        out = text[: len(text) - len(text.lstrip())] + out.strip() + text[len(text.rstrip()):]
        with self._lock:
            self._cache[key] = out
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return out

    def _run(self, text: str, source: str, target: str, context: "str | None",
             history: "list[tuple[str, str]] | None", glossary: "list[tuple[str, str]]",
             form: "str | None") -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class PlamoTranslator(Translator):
    """PLaMo 3 Translate: a PFML3 prompt; subclasses implement `_generate(prompt, max_tokens)`."""

    credit = "Built with PLaMo"

    def _run(self, text, source, target, context, history, glossary, form):
        prompt = pfml3_prompt(text, target, source, context=context, glossary=glossary or None, history=history,
                              form=form)
        return clean(self._generate(prompt, budget(text)))

    def _generate(self, prompt: str, max_tokens: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class HttpTranslator(PlamoTranslator):
    """PLaMo served by an OpenAI-compatible server (the model's bundled serve.py, or vLLM) with /v1/completions."""

    base_url: str = "http://127.0.0.1:8091/v1"
    model: str = "default_model"
    timeout: float = 300.0

    def _generate(self, prompt: str, max_tokens: int) -> str:
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stop": STOP}
        req = urllib.request.Request(self.base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["text"]


LANG_NAMES = {"ja": "Japanese", "en": "English"}
FORMS = {"polite": " Write polite Japanese (です・ます調).", "plain": " Write plain Japanese (だ・である調)."}
CHAT_PROMPT = ("You are a translation engine. Translate the user's text from {source} into {target}. Output only the "
               "translation: no notes, no quotes, no explanation. Keep inline code, math, URLs, Markdown markers, "
               "placeholders such as [[1]], and line breaks exactly as they are.{form}{glossary}{context}")
THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class ChatTranslator(Translator):
    """Any chat model behind an OpenAI-compatible /v1/chat/completions (a local LLM under vLLM, llama.cpp or mlx-lm,
    or a hosted API) asked to translate with an instruction. Earlier paragraph pairs go in as previous turns.
    Thinking is turned off where the server understands `chat_template_kwargs` (Qwen3 and the like) and stripped
    from the output if it comes anyway."""

    base_url: str = "http://127.0.0.1:8091/v1"
    model: str = "default_model"
    api_key: "str | None" = None
    timeout: float = 300.0
    prompt: str = CHAT_PROMPT

    def __post_init__(self) -> None:
        super().__post_init__()
        self._template_kwargs = True  # dropped after a server rejects the field (hosted APIs may)

    def messages(self, text, source, target, context, history, glossary, form) -> list[dict]:
        gl = ("" if not glossary else " Use these term translations: "
              + "; ".join(f"{s} -> {t}" for s, t in glossary) + ".")
        ctx = f" Context (do not translate it): {context}" if context else ""
        system = self.prompt.format(source=LANG_NAMES.get(source, source), target=LANG_NAMES.get(target, target),
                                    form=FORMS.get(form or "", ""), glossary=gl, context=ctx)
        msgs = [{"role": "system", "content": system}]
        for s, t in history or []:
            msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": t}]
        return msgs + [{"role": "user", "content": text}]

    def _run(self, text, source, target, context, history, glossary, form):
        body = {"model": self.model, "messages": self.messages(text, source, target, context, history, glossary, form),
                "max_tokens": budget(text), "temperature": 0}
        if self._template_kwargs:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            j = self._post(body)
        except urllib.error.HTTPError as e:
            if e.code != 400 or "chat_template_kwargs" not in body:
                raise
            self._template_kwargs = False
            del body["chat_template_kwargs"]
            j = self._post(body)
        out = j["choices"][0]["message"].get("content") or ""
        out = THINK.sub("", out)
        return out.split("</think>")[-1]  # a template that opened <think> in the prompt

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)


@dataclass
class MlxTranslator(PlamoTranslator):
    """The MLX-4bit build loaded in this process with mlx-lm (Apple silicon). The model's bundled code is loaded
    (trust_remote_code), as its card says."""

    model_path: str = ""
    wired_gb: float = 2.0
    memory_gb: float = 4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        mx.set_wired_limit(int(self.wired_gb * (1 << 30)))
        mx.set_memory_limit(int(self.memory_gb * (1 << 30)))
        mx.set_cache_limit(1 << 28)
        self._model, self._tok = load(self.model_path, trust_remote_code=True,
                                      tokenizer_config={"trust_remote_code": True})
        self._sampler = make_sampler(temp=0)
        self._gen_lock = threading.Lock()  # one generation at a time on the GPU

    def _generate(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import generate

        with self._gen_lock:
            return generate(self._model, self._tok, prompt=prompt, max_tokens=max_tokens, sampler=self._sampler)


def make_translator(spec: str, kind: str = "plamo", **kw) -> Translator:
    """kind "plamo": `http://host:port/v1` -> HttpTranslator, a local MLX directory -> MlxTranslator.
    kind "chat": any chat model at that URL -> ChatTranslator."""
    remote = spec.startswith(("http://", "https://"))
    if kind == "chat":
        if not remote:
            raise ValueError("a chat translator needs the base URL of an OpenAI-compatible server")
        return ChatTranslator(base_url=spec, **kw)
    if kind != "plamo":
        raise ValueError(f"unknown translator kind {kind!r} (plamo or chat)")
    kw.pop("api_key", None)
    if remote:
        return HttpTranslator(base_url=spec, **kw)
    kw.pop("model", None)
    return MlxTranslator(model_path=spec, **kw)
