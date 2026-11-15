"""Split Markdown into what a translator may touch and what it must not.

Block level: fenced code, display math ($$..$$, \\[..\\], \\begin{env}..\\end{env}) and indented code are never sent
to the translator. Everything else is a paragraph. Inside a paragraph, inline code, inline math and URLs are
protected spans; `translate_inline` tries three ways to keep them byte-for-byte (see there).

`split_blocks` returns pieces whose texts concatenate back to the input exactly, so a caller that translates some
pieces and copies the rest keeps the layout. `BlockStream` does the same for text that arrives in pieces (a
streamed answer): it hands out each block once the block is complete.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator

# A translator for one piece of prose: (text, context or None) -> translated text.
Translate = Callable[[str, "str | None"], str]

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
MATH_BEGIN = re.compile(r"^\s*\\begin\{([A-Za-z*]+)\}")
LIST_MARK = re.compile(r"^(\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?|\s*#{1,6}\s+|\s*(?:>\s?)+)")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")

# Inline spans, in priority order (the first alternative that matches at a position wins).
INLINE = re.compile(
    r"(?P<code>(`+)(?!`).+?(?<!`)\2(?!`))"  # `code`, ``co`de``
    r"|(?P<dmath>\$\$.+?\$\$)"  # $$x$$ inside a line
    r"|(?P<pmath>\\\(.+?\\\))"  # \(x\)
    r"|(?P<bmath>\\\[.+?\\\])"  # \[x\] inside a line
    r"|(?P<math>(?<![\\$\w])\$(?=[^\s$])(?:\\.|[^$\\\n])+?(?<=[^\s\\])\$(?![\w$]))"  # $x$, not $5 and $6 (pandoc's rule)
    r"|(?P<url>https?://[^\s<>()\[\]`]+[^\s<>()\[\]`.,;:!?'\"])"  # a URL, without trailing punctuation
    r"|(?P<tag></?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>)"  # an HTML tag such as <br>
)
CURRENCY = re.compile(r"^\$\d[\d,.]*\$$")  # "$5$" is never math in an answer; keep the pandoc rule from overreaching


@dataclass
class Block:
    kind: str  # "text" (translate), "code", "math" (keep), "blank" (separator, keep)
    text: str
    closed: bool = True  # False for a fence or math block that the text ended inside


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def split_blocks(text: str) -> list[Block]:
    lines = _lines(text)
    out: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        bare = line.rstrip("\r\n")
        if not bare.strip():
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            out.append(Block("blank", "".join(lines[i:j])))
            i = j
            continue
        m = FENCE.match(bare)
        if m:
            mark = m.group(1)
            close = re.compile(r"^ {0,3}" + re.escape(mark[0]) + "{" + str(len(mark)) + r",}\s*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j].rstrip("\r\n")):
                j += 1
            closed = j < len(lines)
            j = min(j + 1, len(lines))
            out.append(Block("code", "".join(lines[i:j]), closed))
            i = j
            continue
        end = _display_math_end(lines, i)
        if end is not None:
            j, closed = end
            out.append(Block("math", "".join(lines[i:j]), closed))
            i = j
            continue
        j = i + 1
        while j < len(lines) and lines[j].strip() and not FENCE.match(lines[j]) and _display_math_end(lines, j) is None:
            j += 1
        para = "".join(lines[i:j])
        kind = "code" if all(l.startswith(("    ", "\t")) for l in lines[i:j]) and not LIST_MARK.match(lines[i]) else "text"
        out.append(Block(kind, para))
        i = j
    return out


def _display_math_end(lines: list[str], i: int) -> "tuple[int, bool] | None":
    """If a display-math block starts at line i, the index just past it and whether it was closed."""
    s = lines[i].strip()
    for opener, closer in (("$$", "$$"), ("\\[", "\\]")):
        if s.startswith(opener):
            rest = s[len(opener):]
            if closer in rest:  # opens and closes on this line: a block only if nothing follows the closer
                return (i + 1, True) if rest.rstrip().endswith(closer) else None
            j = i + 1
            while j < len(lines) and closer not in lines[j]:
                j += 1
            return (j + 1, True) if j < len(lines) else (len(lines), False)
    m = MATH_BEGIN.match(lines[i])
    if m:
        end = "\\end{" + m.group(1) + "}"
        j = i
        while j < len(lines) and end not in lines[j]:
            j += 1
        return (j + 1, True) if j < len(lines) else (len(lines), False)
    return None


class BlockStream:
    """Feed text as it arrives; get each block (or blank separator) once nothing later can change it."""

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, text: str) -> list[Block]:
        self.buf += text
        # Only whole lines are judged (a line of spaces may still get words). The last block can still grow (a
        # paragraph without its blank line yet, an open fence, a blank run that may continue); the ones before it
        # are final.
        cut = self.buf.rfind("\n") + 1
        blocks = split_blocks(self.buf[:cut])
        if len(blocks) < 2:
            return []
        self.buf = blocks[-1].text + self.buf[cut:]
        return blocks[:-1]

    def flush(self) -> list[Block]:
        blocks = split_blocks(self.buf)
        self.buf = ""
        return blocks


def find_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for m in INLINE.finditer(text):
        if m.lastgroup == "math" and CURRENCY.match(m.group(0)):
            continue
        spans.append(m.span())
    return spans


def has_words(text: str) -> bool:
    """Whether there is anything a translator could translate (letters of any script)."""
    return any(ch.isalpha() for ch in text)


def has_prose(text: str) -> bool:
    """Letters outside code, math and URLs."""
    last, rest = 0, []
    for a, b in find_spans(text):
        rest.append(text[last:a])
        last = b
    rest.append(text[last:])
    return has_words("".join(rest))


PLACEHOLDER = "[[{}]]"
STRATEGY = Counter()  # how often each way of keeping spans was used: "as_is", "placeholders", "pieces"


def translate_inline(text: str, translate: Translate, context: "str | None" = None) -> str:
    """Translate one piece of prose, keeping inline code, math and URLs exactly as they were.

    1. Translate the text as it is (the translator sees real code and math, which helps it) and accept the result if
       every protected span is still there, byte for byte.
    2. Otherwise replace the spans with numbered placeholders and accept if each comes back exactly once.
    3. Otherwise translate the prose between the spans one piece at a time (with the whole text as context) and put
       the spans back in their places. Always keeps the spans; the word order may suffer.
    """
    spans = find_spans(text)
    if not has_prose(text):
        return text
    if not spans:
        return translate(text, context)
    originals = [text[a:b] for a, b in spans]
    out = translate(text, context)
    if _all_present(out, originals):
        STRATEGY["as_is"] += 1
        return out
    marks = [PLACEHOLDER.format(k + 1) for k in range(len(spans))]
    masked, last = [], 0
    for (a, b), mark in zip(spans, marks):
        masked += [text[last:a], mark]
        last = b
    masked.append(text[last:])
    out = translate("".join(masked), context)
    if all(out.count(mark) == 1 for mark in marks):
        for mark, orig in zip(marks, originals):
            out = out.replace(mark, orig)
        STRATEGY["placeholders"] += 1
        return out
    pieces, last = [], 0
    for a, b in spans:
        pieces += [_translate_piece(text[last:a], translate, text), text[a:b]]
        last = b
    pieces.append(_translate_piece(text[last:], translate, text))
    STRATEGY["pieces"] += 1
    return "".join(pieces)


def _translate_piece(piece: str, translate: Translate, whole: str) -> str:
    if not has_words(piece):
        return piece
    lead = piece[: len(piece) - len(piece.lstrip())]
    trail = piece[len(piece.rstrip()):]
    return lead + translate(piece.strip(), whole) + trail


def _all_present(out: str, originals: list[str]) -> bool:
    return all(out.count(s) >= n for s, n in Counter(originals).items())


def translate_paragraph(text: str, translate: Translate, context: "str | None" = None) -> str:
    """A paragraph: whole if it is plain prose; line by line (markers kept) if it is a list, headings, quotes or a
    table, so that the structure cannot be lost."""
    body = text.rstrip("\r\n")
    ending = text[len(body):]
    lines = body.split("\n")
    if len(lines) >= 2 and TABLE_SEP.match(lines[1]) and "|" in lines[0]:
        return "\n".join(l if TABLE_SEP.match(l) else _translate_row(l, translate) for l in lines) + ending
    if any(LIST_MARK.match(l) for l in lines):
        out = []
        for l in lines:
            m = LIST_MARK.match(l)
            mark = m.group(0) if m else l[: len(l) - len(l.lstrip())]
            out.append(mark + translate_inline(l[len(mark):], translate, context))
        return "\n".join(out) + ending
    lead = body[: len(body) - len(body.lstrip())]
    return lead + translate_inline(body.strip(), translate, context) + ending


def _translate_row(row: str, translate: Translate) -> str:
    cells = row.split("|")
    return "|".join(c if not has_prose(c) else
                    c[: len(c) - len(c.lstrip())] + translate_inline(c.strip(), translate) + c[len(c.rstrip()):]
                    for c in cells)


def translate_markdown(text: str, translate: Translate, skip: "Callable[[str], bool] | None" = None) -> str:
    """Translate the prose of a Markdown text and copy code, math and layout. `skip(block)` can veto a block (for
    example one that is already in the target language)."""
    return "".join(translate_block(b, translate, skip) for b in split_blocks(text))


def translate_block(block: Block, translate: Translate, skip: "Callable[[str], bool] | None" = None) -> str:
    if block.kind != "text" or not has_prose(block.text) or (skip and skip(block.text)):
        return block.text
    return translate_paragraph(block.text, translate)


def iter_text_blocks(text: str) -> Iterator[str]:
    """The paragraphs a translator would see (for tests and statistics)."""
    for b in split_blocks(text):
        if b.kind == "text":
            yield b.text
