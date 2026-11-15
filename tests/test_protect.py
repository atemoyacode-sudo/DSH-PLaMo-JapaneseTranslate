"""Code, math and layout must survive translation byte for byte, whatever the translator does to them.

  python -m unittest discover -s tests -t .      (from the repository root)
"""

import random
import re
import unittest

from harness.protect import (BlockStream, find_spans, split_blocks, translate_inline, translate_markdown,
                             translate_paragraph)

DOC = r"""Intro line with `x = 1` and $a^2 + b^2 = c^2$ inline.

```python
def f(x):

    return x  # a blank line above, inside the fence
```

$$
\int_0^1 x\,dx = \frac{1}{2}
$$

\begin{align}
a &= b \\
c &= d
\end{align}

\[ e^{i\pi} + 1 = 0 \]

- first item with `code`
- second item
  1. nested one

# A heading

> a quote

| Name | Value |
|---|---|
| speed | 3 m/s |

    indented code
    stays

Costs $5 and $6 today. See https://example.com/a_(b), then \(x+1\) and <br> done."""

PROTECTED = ["`x = 1`", "$a^2 + b^2 = c^2$", "`code`", "https://example.com/a_", r"\(x+1\)", "<br>"]
BLOCKS = ["```python\ndef f(x):\n\n    return x  # a blank line above, inside the fence\n```\n",
          "$$\n\\int_0^1 x\\,dx = \\frac{1}{2}\n$$\n", "\\begin{align}\na &= b \\\\\nc &= d\n\\end{align}\n",
          "\\[ e^{i\\pi} + 1 = 0 \\]\n", "    indented code\n    stays\n", "|---|---|"]


def keeper(text, context=None):
    """A translator that keeps everything it is given (the easy case)."""
    return "T[" + text + "]"


def mangler(text, context=None):
    """A hostile translator: rewrites code, math and URLs, and also breaks [[n]] placeholders."""
    t = re.sub(r"`[^`]*`", "`changed`", text)
    t = re.sub(r"\$[^$]+\$", "$y$", t)
    t = t.replace("https://", "http://").replace("[[", "[").replace("<br>", "<br/>").replace(r"\(", "(")
    return "M[" + t + "]"


def mask_friendly(text, context=None):
    """Rewrites code and math but keeps placeholders (the second strategy should win)."""
    t = re.sub(r"`[^`]*`", "`changed`", text)
    return "P[" + re.sub(r"\$[^$]+\$", "$y$", t) + "]"


class SplitTests(unittest.TestCase):
    def test_round_trip_exact(self):
        for text in (DOC, DOC + "\n", "\n\n" + DOC, "", "one line", "a\r\nb\r\n\r\nc"):
            self.assertEqual("".join(b.text for b in split_blocks(text)), text)

    def test_kinds(self):
        kinds = {b.text: b.kind for b in split_blocks(DOC)}
        for blk in BLOCKS[:5]:
            self.assertIn(blk, kinds, blk)
            self.assertIn(kinds[blk], ("code", "math"))
        self.assertTrue(any(k == "text" and "first item" in t for t, k in kinds.items()))

    def test_unclosed_fence_is_one_block(self):
        b = split_blocks("text\n\n```\ncode\n\nmore code")
        self.assertEqual(b[-1].kind, "code")
        self.assertFalse(b[-1].closed)
        self.assertIn("more code", b[-1].text)

    def test_stream_matches_whole(self):
        rng = random.Random(0)
        for _ in range(200):
            cuts = sorted(rng.sample(range(1, len(DOC)), rng.randint(1, 40)))
            pieces = [DOC[a:b] for a, b in zip([0] + cuts, cuts + [len(DOC)])]
            s, got = BlockStream(), []
            for p in pieces:
                got += s.feed(p)
            got += s.flush()
            self.assertEqual("".join(b.text for b in got), DOC)
            self.assertEqual([(b.kind, b.text) for b in got], [(b.kind, b.text) for b in split_blocks(DOC)])


class SpanTests(unittest.TestCase):
    def test_spans_found(self):
        found = [DOC[a:b] for a, b in find_spans(DOC)]
        for p in PROTECTED:
            self.assertTrue(any(f.startswith(p) for f in found), p)

    def test_currency_is_not_math(self):
        self.assertEqual(find_spans("It costs $5 and $6."), [])
        self.assertEqual(find_spans("between $5$ and $6$"), [])

    def test_math_needs_tight_dollars(self):
        self.assertEqual(find_spans("I have $ 5 and $ 6"), [])
        self.assertEqual([m for m in find_spans("so $x$ is even")], [(3, 6)])


class TranslateTests(unittest.TestCase):
    def check_all_kept(self, out):
        for p in PROTECTED:
            self.assertIn(p, out, p)
        for blk in BLOCKS:
            self.assertIn(blk, out, blk)

    def test_easy_translator(self):
        out = translate_markdown(DOC, keeper)
        self.check_all_kept(out)
        self.assertIn("- T[first item with `code`]", out)
        self.assertIn("# T[A heading]", out)
        self.assertIn("> T[a quote]", out)
        self.assertIn("| T[speed] | T[3 m/s] |", out)
        self.assertNotIn("T[    indented", out)

    def test_hostile_translator_falls_back_to_pieces(self):
        out = translate_markdown(DOC, mangler)
        self.check_all_kept(out)
        self.assertNotIn("`changed`", out)

    def test_placeholders_used_when_they_survive(self):
        out = translate_inline("Use `a()` then $b$.", mask_friendly)
        self.assertEqual(out, "P[Use `a()` then $b$.]")

    def test_three_strategies_in_order(self):
        calls = []

        def spy(text, context=None):
            calls.append(text)
            return mangler(text)

        out = translate_inline("Call `f()` now", spy)
        self.assertEqual(calls[0], "Call `f()` now")  # 1: as is
        self.assertEqual(calls[1], "Call [[1]] now")  # 2: placeholders
        self.assertEqual(calls[2:], ["Call", "now"])  # 3: the pieces
        self.assertEqual(out, "M[Call] `f()` M[now]")

    def test_pieces_get_context(self):
        seen = []
        translate_inline("Call `f()` now", lambda t, c=None: seen.append(c) or mangler(t))
        self.assertEqual(seen[-1], "Call `f()` now")

    def test_layout_whitespace_kept(self):
        self.assertEqual(translate_paragraph("  hello\n", keeper), "  T[hello]\n")
        self.assertEqual(translate_paragraph("- a\n- b", keeper), "- T[a]\n- T[b]")
        self.assertEqual(translate_paragraph("1. a\n2) b\n- [ ] c", keeper), "1. T[a]\n2) T[b]\n- [ ] T[c]")

    def test_nothing_to_translate(self):
        self.assertEqual(translate_markdown("123 + 456\n\n`x`", keeper), "123 + 456\n\n`x`")

    def test_skip(self):
        out = translate_markdown("hello\n\nこんにちは", keeper, skip=lambda b: "こ" in b)
        self.assertEqual(out, "T[hello]\n\nこんにちは")


if __name__ == "__main__":
    unittest.main()
