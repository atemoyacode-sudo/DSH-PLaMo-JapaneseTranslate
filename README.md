# DSH-PLaMo-JapaneseTranslate

Chat in Japanese with a model that is only good in English. Your Japanese message is translated into English with
[PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-0.8b) (Preferred Networks), the model answers in
English, and the answer is translated back into Japanese. Use it from the terminal, or as an OpenAI-compatible proxy
that chat apps and [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh) can point at.

It was built for **Astrantia** (Sharpen-Qwen-3.5-9B-A2.6B-Preview), a 9.4B cut of Qwen3.5-35B-A3B that reasons well
in English but writes weak Japanese: on a 40-prompt language probe it answered in the asked language 2 times out of
40 (Qwen3.5-9B: 40), and its Japanese has broken words. Any model behind an OpenAI-compatible server works.

This is an independent project, not affiliated with or endorsed by DeepSeek or Preferred Networks. It is compatible
with DeepSeek Harness through dsh's own configuration; it is not a dsh plugin.

```
あなた ──日本語──▶ PLaMo (ja→en) ──English──▶ Astrantia ──English──▶ PLaMo (en→ja) ──日本語──▶ あなた
                                               (thinking stays in English, hidden by default)
```

- **Code, math and URLs are never translated.** Fenced code blocks, display math (`$$…$$`, `\[…\]`,
  `\begin{…}…\end{…}`) and indented code are not sent to the translator at all. Inline code, inline math (`$…$`,
  `\(…\)`) and URLs are checked after translation and must come back byte for byte; if PLaMo changed one, the
  paragraph is translated again with numbered placeholders, and if that fails too, the prose between them is
  translated piece by piece (with the whole paragraph as context) so the spans always survive.
- **Layout is kept.** Lists, headings, quotes and tables are translated line by line (or cell by cell) with their
  Markdown markers copied, so the structure cannot be lost. Text that is already in the target language is left as
  it is.
- **Thinking is not translated or shown** by default (`--show-thinking` prints it in English).
- **Answers stream a paragraph at a time**: each block is translated as soon as the model finishes it.
- **The model sees its own English** in a multi-turn chat: the proxy remembers which English answer each Japanese
  answer came from, so history is not a translation of a translation.
- English questions pass through untranslated, and so do their answers.

The harness itself is plain Python (standard library only). PLaMo runs either as its own small server (recommended)
or inside the harness process with mlx-lm.

## Setup (Apple silicon)

Python 3.10 or later. The harness itself uses only the standard library; the two model servers need `mlx-lm`.

1. **The model**: any OpenAI-compatible server. With mlx-lm:

   ```bash
   python -m mlx_lm.server --model <model directory or Hugging Face id> --port 8080
   ```

   vLLM works too (`vllm serve <model> --reasoning-parser qwen3`); pass its URL and served name with `--model-url`
   and `--model-name`.

2. **PLaMo 3 Translate** (MLX-4bit, about 1 GB). It needs `mlx-lm`, `transformers` and `numba` (its tokenizer
   uses numba):

   ```bash
   pip install -U mlx-lm transformers numba huggingface_hub
   ```

   ```bash
   hf download pfnet/plamo-3-translate-0.8b-MLX-4bit --local-dir models/plamo-3-translate-0.8b-MLX-4bit --exclude "figures/*"
   ```

   Serve it with the model's own `serve.py`, under memory limits (from this repository's root):

   ```bash
   python -m harness.serve_plamo --model models/plamo-3-translate-0.8b-MLX-4bit --port 8091
   ```

   This loads the Python code that ships with the model (`trust_remote_code`), as the model card says. PLaMo adds
   about 1 GB of weights next to the main model, within its own memory cap (4 GB by default).

## Chat in the terminal

From this repository's root:

```bash
python -m harness.chat
```

```bash
python -m harness.chat -q "素数が無限にあることを証明してください" --show-english --stats
```

| Option | Default | Meaning |
|---|---|---|
| `--model-url` | `http://127.0.0.1:8080/v1` | The model's OpenAI-compatible base URL |
| `--model-name` | `default_model` | Model name sent to it (mlx-lm: `default_model`; vLLM: the served name) |
| `--translator` | `http://127.0.0.1:8091/v1` | The translator's server URL, or (PLaMo only) a local MLX-4bit directory to load in-process |
| `--translator-kind` | `plamo` | `plamo` (PLaMo 3 Translate) or `chat` (any chat model; see [Other translators](#other-translators)) |
| `--translator-model`, `--translator-api-key` | `default_model`, none | Model name and API key for the translator's server |
| `--no-thinking` | thinking on | Ask the model with thinking off (much faster, a little weaker) |
| `--max-tokens` | 8192 | The model's limit, thinking included |
| `--form` | `polite` | Japanese style of the answers: `polite` (です・ます) or `plain` (だ・である) |
| `--glossary` | none | TSV of `日本語<TAB>English` terms the translator must use, both ways |
| `--show-thinking`, `--show-english`, `--stats` | off | Print the thinking, the English text, the timings |

Every option can also be set by environment variable (`HARNESS_MODEL_URL`, `HARNESS_TRANSLATOR`, ...).
Type `/reset` to clear the history.

## Other translators

PLaMo is the default, but any chat model behind an OpenAI-compatible server can translate instead: a local LLM under
vLLM, llama.cpp or mlx-lm, or a hosted API. It is asked with an instruction (translate, output only the translation,
keep code, math, URLs, Markdown and placeholders), earlier paragraph pairs as previous turns, the glossary, context and
answer style in the instruction. Thinking is turned off for models that support `chat_template_kwargs` (Qwen3 and the
like; servers that reject the field are retried without it), and any `<think>` part is removed from the output.
The same protection applies: code, math and URLs are checked after every translation.

```bash
python -m harness.chat --translator-kind chat --translator http://127.0.0.1:8094/v1 --translator-model <served name>
```

With a hosted API, pass `--translator-api-key` (or `HARNESS_TRANSLATOR_API_KEY`); everything you type and every
answer is then sent to that service. "Built with PLaMo" and the PLaMo model id are shown only while PLaMo translates.

## Use it from a chat app: the proxy

```bash
python -m harness.proxy --port 8092
```

The proxy serves `POST /v1/chat/completions` (streamed or not) and `GET /v1/models` (model id `astrantia-ja-plamo`
with PLaMo, `astrantia-ja` with another translator; `--served-name` to change it), so
any app that talks to OpenAI-compatible servers can point at `http://127.0.0.1:8092/v1`. Japanese user, system and
assistant text is translated for the model; tool definitions, tool calls and tool results pass through unchanged.
The thinking is dropped unless `--pass-thinking` (then it is sent untranslated as `reasoning_content`). Long
thinking keeps the stream alive with SSE comments. The proxy binds to 127.0.0.1 and checks no API key; do not expose
it to a network.

### DeepSeek Harness (dsh)

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) reaches self-hosted OpenAI-compatible servers
through its `@deepseek-ai/dsh-llm-pi-ai` plugin, so it needs no changes: declare a route to the proxy in your
dsh configuration (for example `cordis.patch.yml`).

```yaml
- name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      astrantia-ja-plamo:
        displayName: Astrantia (Japanese, Built with PLaMo)
        api: openai-completions
        baseURL: http://127.0.0.1:8092/v1
        models:
          - id: astrantia-ja-plamo
            name: Astrantia + PLaMo (Japanese)
            contextWindow: 32768
            reasoningEfforts: false  # thinking is set by the proxy (--no-thinking to turn it off)
```

Not tested with dsh yet. dsh is an agent: it runs the shell commands and file edits the model asks for. Keep its
approvals on, and do not let a 9B research preview act on a machine you care about.

## Measured example

2026-11-15, a Mac with an M5 and 32 GB, mlx-lm 0.32.2, both models on the GPU: a pre-release MLX build of Astrantia
(4-bit) and PLaMo 3 Translate 0.8B MLX-4bit. One sample each, so read it as an example, not a benchmark.

Two Japanese prompts, asked straight to the model in Japanese and through the harness, with the same token limits
(1,024 with thinking off, 4,096 with it on):

| Prompt | Thinking | Asked in Japanese directly | Through the harness (total time, of which translating) |
|---|---|---|---|
| Three ways to enjoy a rainy day at home, with reasons | off | Answered in Chinese | Japanese, 513 model tokens (16.8 s, 5.2 s for 2,357 characters) |
| | on | No answer: the thinking ran past 4,096 tokens | Japanese, 1,315 tokens (34.8 s, 3.4 s) |
| 4 bags of 3 apples, 5 eaten: how many are left? | off | Right (7), in mixed Chinese, Japanese and Korean | Right (7), Japanese (4.0 s, 1.1 s) |
| | on | No answer: the thinking ran past 4,096 tokens | Right (7), Japanese (34.7 s, 1.5 s) |

Through the harness the model thinks and answers in English, where it is strong, and in these runs its thinking ended. The
answer to the apples question, thinking off, as the harness printed it:

```
りんごが**7**個残っています。

以下に順を追って説明します：

1.  **リンゴの総数を計算します：**
    袋が4つあり、各袋にリンゴが3個入っています。
    $$4 \text{ bags} \times 3 \text{ apples/bag} = 12 \text{ apples}$$

2.  **食べたリンゴを引く：**
    12個のリンゴから始めて、そのうち5個を食べます。
    $$12 - 5 = 7 \text{ apples}$$

ということで、リンゴが7個残っています。
```

What it does not fix:

- **The model's knowledge.** The rainy-day answers read naturally in Japanese but are still made up ("cook and drink
  rainwater", "go hiking"). Translation changes the language, not the facts.
- **The small translator's slips.** Against the stub model (code, math, a list and a table to keep), every code
  block, formula, inline span, URL and table border came through unchanged (all three paragraphs with inline spans
  kept them on the first try), but PLaMo 0.8B turned "add each square" into 各マス ("each grid square") and the
  question "sum of squares" into "sum of squared". The 2B or 8B build should slip less.
- Math stays as the model wrote it, English words inside `\text{}` included.

The proxy, streamed with thinking off, sent the answer to a LaTeX question in 9 chunks (one per finished block) with
the LaTeX code block and the inline formulas intact.

## Tests

```bash
python -m unittest discover -s tests -t .
```

29 tests, no GPU or model files needed: code, math, tables and layout must survive a translator that keeps them, one
that breaks them, and one that keeps only placeholders; streamed splitting must equal whole-text splitting at any
chunk boundaries; the PFML3 prompt must equal what PLaMo's own chat template renders (run with jinja2 installed and the
model in `models/` or at `PLAMO_DIR`, else skipped); the CLI path and the proxy (stream and non-stream, history, tool calls, an
unreachable model) run end to end against `harness/stub_model.py`, a tiny stand-in for the model's server; the chat
translator's request (instruction, glossary, history, thinking off) and its fallback for servers that reject
`chat_template_kwargs` are checked against a stub.

## Limits

- Meaning that lives in the Japanese words themselves (wordplay, kanji readings, "how do you say 猫 in English?")
  can be lost in translation before the model sees it.
- The 0.8B translator is fast but small; PFN's 2B and 8B builds translate better (same prompt format, set
  `--translator` to a server running one of them). A chat model as translator is untested here and may add notes
  or answer the text instead of translating it.
- Tool results and other text the model reads are not translated.
- Images and other non-text content pass through untouched.

## License

The harness code is under the [MIT License](LICENSE). **PLaMo 3 Translate is not covered by it**: the model is under
the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en), and this harness does
not ship it (you download it from PFN). In short:

- Personal use (learning, hobby projects) and other non-commercial use, including academic use, are free with no
  registration.
- Commercial use (which the license defines broadly: business use, and building or offering products, services or
  APIs) is free when the latest annual revenue of the user and its affiliates is 1 billion yen or less, after
  registering through PFN's form; above that it needs a separate agreement with PFN.
- Anything provided to others that uses the model must show "Built with PLaMo", and an AI model created with PLaMo
  must have "PLaMo" in its name. This harness does both while PLaMo translates: "Built with PLaMo" in the chat
  banner, in the proxy's model list and here, and the proxy's model id `astrantia-ja-plamo`.

Read the license itself before relying on this summary.

Built with PLaMo.
