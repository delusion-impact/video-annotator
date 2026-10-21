"""Построение аннотации по транскрипту (локальная экстрактивная суммаризация).

Уровни детализации (--detail):
  1 — максимально кратко (1 абзац, ~60 слов)
  2 — кратко (~150 слов)
  3 — умеренно (~300 слов, тезисы + ключевые слова)
  4 — подробно (~600 слов, разделы по темам/таймкодам)
  5 — детально (~1000+ слов, все темы с подтемами и примерами)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .ui import fmt_duration

DETAIL_PROFILES = {
    1: {"name": "максимально кратко", "words": 60, "sections": 0, "quotes": 0,
        "keywords": 6, "intro": True, "bullets": False},
    2: {"name": "кратко", "words": 150, "sections": 0, "quotes": 0,
        "keywords": 8, "intro": True, "bullets": True},
    3: {"name": "умеренно", "words": 300, "sections": 2, "quotes": 0,
        "keywords": 10, "intro": True, "bullets": True},
    4: {"name": "подробно", "words": 600, "sections": 4, "quotes": 2,
        "keywords": 15, "intro": True, "bullets": True},
    5: {"name": "детально", "words": 1000, "sections": 7, "quotes": 3,
        "keywords": 20, "intro": True, "bullets": True},
}

STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между это
would could should there their about which when while into other than then them
they these those from been have has had does did will your you our out up if
about all any each few more most other some such no nor own same so too very
can just now with who what where how why only also over under after before
""".split())

_WORD_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)
_SENT_END_RE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")
MAX_SENTENCE_WORDS = 35


@dataclass
class Sentence:
    text: str
    start: float
    words: list[str] = field(default_factory=list)
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _split_sentences(text: str, start: float) -> list[Sentence]:
    out = []
    for part in _SENT_END_RE.split(text.strip()):
        part = part.strip()
        if len(part) < 15:
            continue
        # длинные «предложения» (речь без знаков препинания) режем по запятым
        if len(part.split()) > MAX_SENTENCE_WORDS:
            parts = _CLAUSE_RE.split(part)
        else:
            parts = [part]
        for p in parts:
            p = p.strip()
            if len(p) < 15:
                continue
            sent = Sentence(text=p, start=start, words=_tokenize(p))
            sent.words = [w for w in sent.words if w not in STOPWORDS and len(w) > 3]
            out.append(sent)
    return out


def _merge_segments(segments: list[dict]) -> list[tuple[str, float]]:
    """Склеивает сегменты whisper до границ предложений.

    Сегменты whisper — обрывки, режущие фразы посреди слова. Склеиваем их
    до точки/знака вопроса, чтобы предложения были цельными.
    """
    chunks: list[tuple[str, float]] = []
    buf: list[str] = []
    buf_start = segments[0]["start"] if segments else 0.0

    for seg in segments:
        if not buf:
            buf_start = seg["start"]
        buf.append(seg["text"].strip())
        joined = " ".join(buf)
        if _SENT_END_RE.search(joined[-160:]):
            chunks.append((joined, buf_start))
            buf = []
    if buf:
        chunks.append((" ".join(buf), buf_start))
    return chunks


def build_sentences(segments: list[dict]) -> list[Sentence]:
    sentences: list[Sentence] = []
    for text, start in _merge_segments(segments):
        sentences.extend(_split_sentences(text, start))
    return [s for s in sentences if s.words]


def _tfidf(sentences: list[Sentence]) -> dict[str, float]:
    n = len(sentences)
    df: Counter[str] = Counter()
    for s in sentences:
        df.update(set(s.words))
    return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}


CONTINUATION_STARTS = (
    "и ", "а ", "но ", "то есть", "вот ", "ну ", "да ", "и вот ",
    "потому", "поэтому ", "кстати",
)

MIN_CONTENT_WORDS = 4


def _is_substantive(s: Sentence) -> bool:
    """Отсеивает обрывки и продолжения мысли без содержательных слов."""
    if len(s.words) < MIN_CONTENT_WORDS:
        return False
    low = s.text.lower().lstrip("—–- ")
    return not low.startswith(CONTINUATION_STARTS)


def _score(sentences: list[Sentence], idf: dict[str, float], duration: float) -> None:
    freq: Counter[str] = Counter()
    for s in sentences:
        freq.update(s.words)
    max_tf = max(freq.values(), default=1)

    for i, s in enumerate(sentences):
        # TF-IDF: редкие содержательные слова важнее
        score = sum(idf.get(w, 1.0) * (freq[w] / max_tf) for w in set(s.words))
        score /= math.sqrt(len(s.words) + 1)
        # позиционный бонус: вступление и итог обычно содержат главную мысль
        if s.start < duration * 0.05:
            score *= 1.35
        if duration - s.start < duration * 0.07:
            score *= 1.25
        # абзацный бонус: короткие предложения рядом с длинными — контекст
        if 5 <= len(s.words) <= 30:
            score *= 1.1
        s.score = score


def _mmr_select(candidates: list[Sentence], limit: int, lam: float = 0.7) -> list[Sentence]:
    """Maximal Marginal Relevance: баланс релевантности и неповторяемости."""
    if limit <= 0:
        return []
    selected: list[Sentence] = []
    pool = sorted(candidates, key=lambda s: s.score, reverse=True)
    remaining = pool[:400]  # кандидатов достаточно из топа
    while remaining and len(selected) < limit:
        best, best_val = None, -1e18
        for s in remaining:
            if selected:
                overlap = max(
                    len(set(s.words) & set(t.words)) / max(1, len(set(s.words) | set(t.words)))
                    for t in selected
                )
            else:
                overlap = 0.0
            val = lam * s.score - (1 - lam) * overlap * 100.0
            if val > best_val:
                best, best_val = s, val
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    selected.sort(key=lambda s: s.start)
    return selected


def _keywords(sentences: list[Sentence], idf: dict[str, float], limit: int) -> list[str]:
    weights: Counter[str] = Counter()
    for s in sentences:
        for w in set(s.words):
            weights[w] += idf.get(w, 1.0)
    # убираем общеупотребимые, даже если не в стоп-листе
    top = weights.most_common(limit * 3)
    picked: list[str] = []
    for w, _ in top:
        if any(w in p or p in w for p in picked):
            continue
        picked.append(w)
        if len(picked) >= limit:
            break
    return picked


def _sections(sentences: list[Sentence], idf: dict[str, float],
              count: int, duration: float,
              used: set[int] | None = None) -> list[tuple[float, float, str]]:
    """Делит хронологию на блоки и даёт заголовок-аннотацию каждому."""
    if count <= 0 or not sentences:
        return []
    used = used or set()
    bounds = [duration * i / count for i in range(count + 1)]
    out = []
    for i in range(count):
        block = [s for s in sentences if bounds[i] <= s.start < bounds[i + 1]]
        if not block:
            continue
        good = [s for s in block if _is_substantive(s) and id(s) not in used]
        if not good:
            good = [s for s in block if id(s) not in used] or block
        chosen = _mmr_select(good, limit=3)
        text = _join(chosen, 120)
        out.append((bounds[i], bounds[i + 1], text))
    return out


def _shorten(text: str, words: int) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + "…"


def _join(sentences: list[Sentence], budget_words: int) -> str:
    """Склеивает предложения целиком, укладываясь в бюджет слов."""
    out: list[str] = []
    used = 0
    for s in sentences:
        n = len(s.text.split())
        if used and used + n > budget_words:
            break
        # даже первое предложение ограничиваем бюджетом
        text = s.text.rstrip(".!?… ")
        if n > budget_words:
            text = " ".join(text.split()[:budget_words])
        out.append(text + ".")
        used += min(n, budget_words)
        if used >= budget_words:
            break
    return " ".join(out)


def build_annotation(
    segments: list[dict],
    detail: int,
    duration: float,
    title: str,
) -> str:
    """Собирает текстовую аннотацию заданной степени детализации."""
    profile = DETAIL_PROFILES[detail]
    sentences = build_sentences(segments)
    if not sentences:
        raise ValueError("нет пригодных предложений для аннотации")

    idf = _tfidf(sentences)
    _score(sentences, idf, duration)

    budget = profile["words"]
    total_words = sum(len(s.text.split()) for s in sentences)

    lines: list[str] = []
    lines.append(f"АННОТАЦИЯ: {title}")
    lines.append(
        f"Длительность: {fmt_duration(duration)} | "
        f"Детализация: {detail} ({profile['name']}) | "
        f"Объём транскрипта: ~{total_words:,} слов".replace(",", " ")
    )
    lines.append("")

    # 1. Краткое содержание (не более 6 предложений, иначе съедает тезисы)
    intro_budget = min(150, max(40, budget // 4))
    intro_sents = _mmr_select(
        [s for s in sentences if _is_substantive(s)] or sentences,
        limit=max(2, min(6, budget // 120)),
    )
    lines.append("== КРАТКОЕ СОДЕРЖАНИЕ ==")
    lines.append(_join(intro_sents, intro_budget))
    lines.append("")

    # 2. Основные тезисы (для уровней 2+)
    bullets: list[Sentence] = []
    if profile["bullets"]:
        bullet_budget = int(budget * 0.55)
        n_bullets = max(3, min(12, bullet_budget // 45))
        pool = [s for s in sentences if _is_substantive(s)]
        chosen = _mmr_select(pool, limit=n_bullets + len(intro_sents))
        bullets = [s for s in chosen if s not in intro_sents][:n_bullets]
        if bullets:
            lines.append("== ОСНОВНЫЕ ТЕЗИСЫ ==")
            for s in bullets:
                prefix = f"[{fmt_duration(s.start)}] " if profile["sections"] >= 3 else ""
                lines.append(f"- {prefix}{_shorten(s.text, 40)}")
            lines.append("")

    # 3. Разделы по хронологии (уровни 3+) — на предложениях, не взятых в тезисы
    used = set(id(s) for s in intro_sents)
    if profile["bullets"]:
        used |= set(id(s) for s in bullets)
    sections = _sections(sentences, idf, profile["sections"], duration, used)
    if sections:
        lines.append("== РАЗДЕЛЫ ПО ХРОНОЛОГИИ ==")
        for start, end, text in sections:
            lines.append(f"[{fmt_duration(start)} – {fmt_duration(end)}] "
                         f"{_shorten(text, max(45, budget // max(1, len(sections))))}")
        lines.append("")

    # 4. Ключевые слова
    kws = _keywords(sentences, idf, profile["keywords"])
    if kws:
        lines.append("== КЛЮЧЕВЫЕ СЛОВА ==")
        lines.append(", ".join(kws))
        lines.append("")

    # 5. Цитаты (уровни 4+)
    if profile["quotes"]:
        pool = [s for s in sentences if 8 <= len(s.words) <= 28]
        if not pool:
            # нет предложений подходящей длины — берём любые содержательные
            pool = [s for s in sentences if _is_substantive(s)]
        quotes = _mmr_select(pool, limit=profile["quotes"])
        if quotes:
            lines.append("== ЦИТАТЫ ==")
            for s in quotes:
                lines.append(f'- "{_shorten(s.text, 25)}" ({fmt_duration(s.start)})')
            lines.append("")

    return "\n".join(lines).strip() + "\n"
