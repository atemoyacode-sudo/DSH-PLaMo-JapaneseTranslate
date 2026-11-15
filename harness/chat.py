"""Chat with Astrantia in Japanese from the terminal.

  python -m harness.chat                                  # interactive
  python -m harness.chat -q "素数が無限にあることを証明して"   # one question, then exit
  python -m harness.chat --show-english --stats            # also print the English the model saw and wrote

The model is any OpenAI-compatible endpoint (--model-url). The translator is PLaMo 3 Translate (a server URL, or a
local MLX-4bit directory loaded in this process), or with --translator-kind chat any chat model at a server URL.
"""

from __future__ import annotations

import argparse
import os
import sys

from .pipeline import Harness, UpstreamError
from .protect import STRATEGY
from .translator import make_translator

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def add_common(ap: argparse.ArgumentParser) -> None:
    env = os.environ.get
    ap.add_argument("--model-url", default=env("HARNESS_MODEL_URL", "http://127.0.0.1:8080/v1"),
                    help="the model's OpenAI-compatible base URL (vLLM, mlx-lm server)")
    ap.add_argument("--model-name", default=env("HARNESS_MODEL_NAME", "default_model"),
                    help="model name sent to that server (mlx-lm: default_model; vLLM: the served name)")
    ap.add_argument("--api-key", default=env("HARNESS_MODEL_API_KEY"))
    ap.add_argument("--translator", default=env("HARNESS_TRANSLATOR", "http://127.0.0.1:8091/v1"),
                    help="the translator: a server base URL, or (PLaMo only) a local MLX-4bit model directory")
    ap.add_argument("--translator-kind", choices=["plamo", "chat"], default=env("HARNESS_TRANSLATOR_KIND", "plamo"),
                    help="plamo: PLaMo 3 Translate (PFML3 prompts); chat: any chat model, asked to translate")
    ap.add_argument("--translator-model", default=env("HARNESS_TRANSLATOR_MODEL", "default_model"),
                    help="model name sent to the translator server")
    ap.add_argument("--translator-api-key", default=env("HARNESS_TRANSLATOR_API_KEY"),
                    help="API key for a hosted chat translator (your text is sent to that service)")
    ap.add_argument("--no-thinking", action="store_true", help="ask the model with thinking off (faster)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--form", choices=["polite", "plain"], default="polite", help="Japanese style of the answers")
    ap.add_argument("--glossary", help="TSV file of Japanese<TAB>English term pairs for PLaMo")


def build(args: argparse.Namespace) -> Harness:
    glossary = []
    if args.glossary:
        with open(args.glossary, encoding="utf-8") as f:
            glossary = [tuple(l.rstrip("\n").split("\t")[:2]) for l in f if "\t" in l and not l.startswith("#")]
    tr = make_translator(args.translator, args.translator_kind, form=args.form, glossary=glossary,
                         model=args.translator_model, api_key=args.translator_api_key)
    return Harness(tr, base_url=args.model_url, model=args.model_name, api_key=args.api_key,
                   thinking=not args.no_thinking, max_tokens=args.max_tokens)


def ask(h: Harness, history: list[dict], question: str, args: argparse.Namespace) -> None:
    msgs, target = h.prepare([{"role": "user", "content": question}])
    english_q = msgs[0]["content"]
    if args.show_english:
        print(f"{DIM}[en →] {english_q}{RESET}")
    history.append({"role": "user", "content": english_q})
    thinking_open = False
    for ev in h.run(history, target):
        if ev.kind == "reasoning" and args.show_thinking:
            if not thinking_open:
                print(f"{DIM}[thinking] ", end="")
                thinking_open = True
            print(ev.text, end="", flush=True)
        elif ev.kind == "text":
            if thinking_open:
                print(RESET)
                thinking_open = False
            print(ev.text, end="", flush=True)
        elif ev.kind == "tool_calls":
            print(f"\n{DIM}[tool calls, not run] {ev.data}{RESET}")
        elif ev.kind == "finish":
            if thinking_open:
                print(RESET)
            d = ev.data
            if not d["english"].strip():
                print("(回答がありませんでした: 思考の途中でトークン上限に達した可能性があります。--max-tokens を増やすか --no-thinking を試してください)")
            print()
            if args.show_english:
                print(f"{DIM}[← en] {d['english'].strip()}{RESET}")
            if args.stats:
                s, u = h.stats, d.get("usage") or {}
                print(f"{DIM}[stats] total {s['total_s']:.1f}s, translating the answer {s['translate_s']:.1f}s "
                      f"({s['translated_chars']} chars), model tokens {u.get('completion_tokens', '?')}, "
                      f"finish {d['finish_reason']}, spans kept by {dict(STRATEGY) or '-'}{RESET}")
            history.append({"role": "assistant", "content": d["english"]})


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap)
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--show-thinking", action="store_true", help="print the model's thinking (English, untranslated)")
    ap.add_argument("--show-english", action="store_true", help="print the English question and answer too")
    ap.add_argument("--stats", action="store_true", help="print timings after each answer")
    args = ap.parse_args(argv)
    h = build(args)
    history: list[dict] = []
    try:
        if args.question:
            if h.translator.credit:
                print(f"{DIM}({h.translator.credit}){RESET}", file=sys.stderr)
            ask(h, history, args.question, args)
            return 0
        credit = f", {h.translator.credit}" if h.translator.credit else ""
        print(f"Astrantia (日本語ハーネス{credit})。終了は Ctrl-D、履歴の消去は /reset。")
        while True:
            try:
                q = input("\nあなた> ").strip()
            except EOFError:
                print()
                return 0
            if not q:
                continue
            if q == "/reset":
                history.clear()
                print("履歴を消しました。")
                continue
            print("モデル> ", end="", flush=True)
            try:
                ask(h, history, q, args)
            except UpstreamError as e:
                print(f"\nエラー: {e}", file=sys.stderr)
                history.pop()  # the question that got no answer
    except UpstreamError as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    except BrokenPipeError:  # output piped into head and the like
        sys.stderr.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
