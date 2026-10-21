"""Абстрактивный реферат (аннотация) по алгоритму подготовки пересказа.

Реализованы пять шагов:
  1. Первичное сканирование — определить тему, проблему и структуру.
  2. Активное чтение и маркировка — извлечь тезисы по блокам, отбросить
     примеры, не иллюстрирующие главный тезис.
  3. Фильтрация 80/20 — убрать повторы, окраску, второстепенные детали.
  4. Написание черновика своими словами со структурой «введение — аргументы —
     заключение», зеркалящей исходный материал.
  5. Редактирование и нейтральность — убрать оценки, проверить переходы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLMClient
from .ui import fmt_duration

# Целевой объём и структура по уровням детализации
PROFILES: dict[int, dict] = {
    1: {"words": 200, "args": 3, "sections": 0, "quotes": 0,
        "structure": False, "own_words": True},
    2: {"words": 400, "args": 5, "sections": 0, "quotes": 0,
        "structure": True, "own_words": True},
    3: {"words": 700, "args": 7, "sections": 0, "quotes": 0,
        "structure": True, "own_words": True},
    4: {"words": 1100, "args": 9, "sections": 4, "quotes": 0,
        "structure": True, "own_words": True},
    5: {"words": 1500, "args": 11, "sections": 6, "quotes": 3,
        "structure": True, "own_words": True},
}

CHUNK_WORDS = 2000  # размер блока для пошагового разбора

# Слова-маркеры тезисных утверждений (шаг 2 рекомендаций)
THESIS_MARKERS = (
    "следовательно", "таким образом", "несмотря на то", "ключевой момент",
    "главное", "важно", "нужно", "необходимо", "первое", "второе",
    "вывод", "итог", "то есть", "поэтому",
)

# Разговорные/риторические обороты, которые вычищаются на шаге 3
FILLER_RE = re.compile(
    r"(?<!\w)(ну|вот|как бы|как-будто|то есть типа|это самое|в общем|"
    r"по большому счету|так сказать|значит вот|и вот)(?!\w)",
    re.IGNORECASE,
)

# Обращения к залу и вопросы слушателей — не содержание лекции
AUDIENCE_RE = re.compile(
    r"^\s*([А-ЯЁ][а-яё]{2,15})\s*,\s*.*\?|^.*\?\s*$"
)


@dataclass
class Chunk:
    """Блок транскрипта для пошагового разбора."""
    index: int
    start: float
    end: float
    text: str


def _clean(text: str) -> str:
    """Шаг 3: убрать риторику и разговорные обороты."""
    text = FILLER_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_chunks(segments: list[dict], chunk_words: int = CHUNK_WORDS) -> list[Chunk]:
    """Делит транскрипт на блоки по объёму, сохраняя таймкоды."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = segments[0]["start"] if segments else 0.0
    buf_end = buf_start
    words = 0

    for seg in segments:
        if not buf:
            buf_start = seg["start"]
        buf.append(seg["text"].strip())
        buf_end = seg["end"]
        words += len(seg["text"].split())
        if words >= chunk_words:
            chunks.append(Chunk(len(chunks), buf_start, buf_end, " ".join(buf)))
            buf, words = [], 0

    if buf:
        chunks.append(Chunk(len(chunks), buf_start, buf_end, " ".join(buf)))
    return chunks


def _format_segments(segments: list[dict], limit: int = 4000) -> str:
    """Фрагмент транскрипта с таймкодами (для сканирования)."""
    out = []
    for seg in segments[:limit]:
        out.append(f"[{fmt_duration(seg['start'])}] {seg['text'].strip()}")
    return "\n".join(out)


# --------------------------------------------------------------------------
# ШАГ 1. Первичное сканирование
# --------------------------------------------------------------------------
STEP1_PROMPT = """Ты — методист, готовящий реферат по записи лекции.

ЗАДАЧА (первичное сканирование): определи структуру материала, не читая его
подряд, а по опорным точкам: начало, конец и оглавление тем.

Начало записи:
{head}

Конец записи:
{tail}

Дай ответ СТРОГО в таком виде:
ТЕМА: <тема одним предложением>
ПРОБЛЕМА: <какая задача/вопрос решается — одно предложение>
ЦЕЛЬ: <цель автора — одно предложение>
СТРУКТУРА: <через запятую перечисли основные разделы/темы материала по порядку>
КЛЮЧЕВЫЕ ПОНЯТИЯ: <5-10 терминов через запятую>
ИТОГИ: <1-2 предложения о выводах автора, если они ясны из конца записи>

Не выдумывай того, чего нет в тексте. Пиши по-русски, без вступлений."""


def step1_preview(client: LLMClient, segments: list[dict]) -> str:
    """Шаг 1: возвращает разбор структуры материала."""
    head = _format_segments(segments[:120])
    tail = _format_segments(segments[-120:])
    return client.complete(STEP1_PROMPT.format(head=head, tail=tail))


# --------------------------------------------------------------------------
# ШАГИ 2-3. Маркировка и фильтрация по блокам
# --------------------------------------------------------------------------
STEP2_PROMPT = """Ты — методист, готовящий реферат по записи лекции.

ЗАДАЧА: разбери блок записи и выпиши ТОЛЬКО содержательные тезисы.

Блок №{num} (таймкод {start}–{end}):

{text}

ПРАВИЛА ОТБОРА (строго):
- Тезис — утверждение автора, которое несёт основной смысл: правило,
  рекомендация, определение, вывод, ответ на проблему.
- Признак тезиса: связующие слова («следовательно», «таким образом»,
  «несмотря на то что», «ключевой момент», «главное —», «первое/второе»),
  либо это первая фраза тематического абзаца.
- ОТБРАСЫВАЙ: примеры, бытовые отступления, повторы, личные истории,
  риторику, обращения к залу, вопросы слушателей и ответы на них,
  эмоциональные оценки, второстепенные детали и ссылки на другие работы.
- Если пример важен, передай его суть ОДНИМ предложением в тезис.
- Каждый тезис формулируй СВОИМИ СЛОВАМИ, одним-двумя предложениями.
  НЕ КОПИРУЙ предложения из текста дословно.
- Указывай таймкод, если можешь понять его из пометок в тексте блока.

ФОРМАТ ОТВЕТА (строго, каждый тезис — отдельной строкой, с дефисом в начале,
без нумерации и без квадратных скобок):
- <тезис своими словами>
- <тезис своими словами>
...
Если содержательных тезисов в блоке нет, ответь одной строкой: НЕТ ТЕЗИСОВ"""


# Маркеры списка, которые может использовать модель
LIST_MARKERS = ("-", "*", "•", "—", "–", "[")
NUMBERED_RE = re.compile(r"^\d{1,2}[.)]\s+")


def _parse_theses(raw: str) -> list[str]:
    """Разбирает ответ модели в список тезисов.

    Модель может оформлять их дефисом, звёздочкой, нумерацией или квадратными
    скобками — принимаем все варианты, а если маркеров нет вовсе, берём
    непустые строки целиком.
    """
    marked: list[str] = []
    plain: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "НЕТ ТЕЗИСОВ" in line.upper():
            return []
        if NUMBERED_RE.match(line):
            item = NUMBERED_RE.sub("", line).strip(" .")
            if len(item) >= 15:
                marked.append(item)
        elif line[0] in LIST_MARKERS:
            item = line.lstrip("".join(LIST_MARKERS)).strip(" \t[]*•—–-")
            item = item.strip(" .")
            if len(item) >= 15:
                marked.append(item)
        elif len(line) >= 25 and not line.startswith(("#", "=")):
            plain.append(line.strip(" ."))

    theses = marked or plain
    return [t for t in theses if len(t) >= 15][:12]


def step2_extract(client: LLMClient, chunk: Chunk) -> list[str]:
    """Шаги 2-3: возвращает список тезисов блока."""
    prompt = STEP2_PROMPT.format(
        num=chunk.index + 1,
        start=fmt_duration(chunk.start),
        end=fmt_duration(chunk.end),
        text=_clean(chunk.text)[:12000],
    )
    raw = client.complete(prompt)
    return _parse_theses(raw)


# --------------------------------------------------------------------------
# ШАГ 4. Написание черновика
# --------------------------------------------------------------------------
STEP4_PROMPT = """Ты — методист, пишущий РЕФЕРАТ (пересказ) по конспекту лекции.

РАЗБОР МАТЕРИАЛА:
{preview}

ИЗВЛЕЧЁННЫЕ ТЕЗИСЫ ПО БЛОКАМ:
{theses}

НАПИШИ РЕФЕРАТ. Требования:

1. СТРУКТУРА (если она указана):
   ## ТЕМА И ПРОБЛЕМА
   1-2 предложения: о чём материал и какую задачу решает.
   ## ОСНОВНЫЕ АРГУМЕНТЫ
   Нумерованный список {n_args} пунктов. Каждый пункт — 1-3 предложения,
   сформулированные СВОИМИ СЛОВАМИ. Логика пунктов повторяет ход рассуждения
   автора: тезис → следствие → практический вывод.
   ## ВЫВОДЫ И РЕКОМЕНДАЦИИ
   Итог и рекомендации автора.
{extra}
2. ЯЗЫК: пиши СВОИМИ СЛОВАМИ, не копируй предложения из конспекта.
   Разрешённые переформулировки по смыслу, но не дословные цитаты.
3. НЕЙТРАЛЬНОСТЬ: передавай позицию автора. Никаких «я думаю», «мне кажется»,
   «по-моему», оценочных прилагательных от себя.
4. ЛОГИКА: причинно-следственные связи должны быть понятны читателю.
   Используй связки: «поэтому», «следовательно», «таким образом», «поскольку».
5. ОБЪЁМ: примерно {words} слов. Не выходи за рамки исходного материала,
   не добавляй фактов, которых в тезисах нет.
6. Не добавляй заголовок с названием файла, вступление и комментарии «вот
   краткое содержание» — только сам реферат.

Если в разделе «ОСНОВНЫЕ АРГУМЕНТЫ» уместны подзаголовки по темам, добавь их
строками вида «### Тема» перед соответствующими пунктами.

Выведи только текст реферата."""


def step4_draft(
    client: LLMClient,
    preview: str,
    theses_by_chunk: list[tuple[Chunk, list[str]]],
    detail: int,
    sections: list[tuple[float, float, str]],
) -> str:
    """Шаг 4: пишет черновик реферата."""
    profile = PROFILES[detail]

    thesis_lines: list[str] = []
    for chunk, theses in theses_by_chunk:
        if not theses:
            continue
        thesis_lines.append(f"Блок {fmt_duration(chunk.start)}–{fmt_duration(chunk.end)}:")
        thesis_lines.extend(f"- {t}" for t in theses)
        thesis_lines.append("")

    extra: list[str] = []
    if profile["sections"] and sections:
        extra.append("Разделы исходного материала (для подзаголовков):\n")
        for start, end, theme in sections:
            extra.append(f"- {fmt_duration(start)}–{fmt_duration(end)}: {theme}\n")
        extra.append("\n")
    if profile["quotes"]:
        extra.append(
            "После раздела «ВЫВОДЫ И РЕКОМЕНДАЦИИ» НЕ добавляй раздел с цитатами —\n"
            "дословные цитаты будут добавлены отдельно.\n\n"
        )
    if not profile["structure"]:
        extra.append(
            "Структуру с подзаголовками НЕ используй: дай сплошной связный\n"
            "текст на 2-3 абзаца без заголовков.\n\n"
        )

    prompt = STEP4_PROMPT.format(
        preview=preview,
        theses="\n".join(thesis_lines)[:24000],
        n_args=profile["args"],
        words=profile["words"],
        extra="".join(extra),
    )
    return client.complete(prompt)


# --------------------------------------------------------------------------
# ШАГ 5. Редактирование и нейтральность
# --------------------------------------------------------------------------
STEP5_PROMPT = """Ты — редактор. Ниже черновик реферата, составленного по записи лекции.

ЧЕРНОВИК:
{draft}

ЗАДАЧА (редактура):
1. НЕЙТРАЛЬНОСТЬ: убери любые оценки от редактора («по-моему», «это отличный
   совет», «важно отметить»). Реферат должен передавать позицию АВТОРА.
2. ЛОГИКА: проверь причинно-следственные связи. Там, где переход неочевиден,
   добавь связку («поэтому», «следовательно», «поскольку»). Не переписывай
   смысл, только связывай.
3. ЦЕЛОСТНОСТЬ: убери повторы, «шероховатые» переходы и повторное введение
   одной мысли разными словами.
4. ЯЗЫК: убери слова-паразиты и разговорные обороты, вычисти пунктуацию.
5. СТИЛЬ: ровный, без оценочных прилагательных, без восклицаний.
6. Сохрани объём (±10%) и структуру с подзаголовками, если они были.

Если черновик уже в порядке — верни его без изменений, без комментариев.
Выведи только итоговый текст реферата."""


def step5_edit(client: LLMClient, draft: str) -> str:
    """Шаг 5: редактура черновика."""
    return client.complete(STEP5_PROMPT.format(draft=draft[:24000]))


# --------------------------------------------------------------------------
# Сборка итогового реферата
# --------------------------------------------------------------------------
def _extract_themes(preview: str, limit: int) -> list[str]:
    """Достаёт перечисление тем из разбора структуры (шаг 1)."""
    m = re.search(r"СТРУКТУРА:\s*(.+)", preview)
    if not m:
        return []
    raw = m.group(1).strip()
    parts = re.split(r"[,;]|\d+\.\s*", raw)
    return [p.strip(" .—–-") for p in parts if len(p.strip(" .—–-")) > 3][:limit]


def _themes_with_time(
    client: LLMClient,
    segments: list[dict],
    preview: str,
    count: int,
) -> list[tuple[float, float, str]]:
    """Сопоставляет темы из шага 1 с таймкодами по блокам."""
    chunks = split_chunks(segments)
    if not chunks or count <= 0:
        return []
    themes = _extract_themes(preview, count)
    if not themes:
        return []

    # ищем блок, где тема упоминается чаще всего
    out: list[tuple[float, float, str]] = []
    used: set[int] = set()
    for theme in themes:
        key = theme.lower()[:12]
        best_idx, best_hits = -1, 0
        for i, ch in enumerate(chunks):
            if i in used:
                continue
            hits = ch.text.lower().count(key)
            if hits > best_hits:
                best_idx, best_hits = i, hits
        if best_idx >= 0:
            used.add(best_idx)
            out.append((chunks[best_idx].start, chunks[best_idx].end, theme))
    return out


def _extract_quotes(segments: list[dict], limit: int) -> list[tuple[float, str]]:
    """Выбирает дословные цитаты прямо из транскрипта.

    Цитаты берутся из исходника, а не из ответа модели: так они
    действительно дословные и не выдуманы.
    """
    from .summary import build_sentences, _tfidf, _score, _mmr_select, _is_substantive

    sentences = build_sentences(segments)
    if not sentences:
        return []
    idf = _tfidf(sentences)
    _score(sentences, idf, segments[-1]["end"] if segments else 0.0)
    pool = [s for s in sentences if _is_substantive(s) and 8 <= len(s.words) <= 26]
    if not pool:
        pool = [s for s in sentences if _is_substantive(s)]
    chosen = _mmr_select(pool, limit=limit)
    return [(s.start, s.text) for s in chosen]


# --------------------------------------------------------------------------
# Кэш промежуточных результатов LLM
# --------------------------------------------------------------------------
def _cache_key(segments: list[dict], detail: int) -> str:
    """Ключ кэша: состав транскрипта + уровень детализации."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(detail).encode())
    for seg in segments[::50]:
        h.update(f"{seg['start']:.1f}{seg['text']}".encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _load_cache(path: str, key: str) -> dict | None:
    import json
    from pathlib import Path as _P
    try:
        data = json.loads(_P(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("key") != key:
        return None
    return data


def _save_cache(path: str, key: str, payload: dict) -> None:
    import json
    from pathlib import Path as _P
    try:
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"key": key, **payload}, ensure_ascii=False),
                     encoding="utf-8")
    except OSError:
        pass


def build_llm_annotation(
    client: LLMClient,
    segments: list[dict],
    detail: int,
    duration: float,
    title: str,
    stage=None,
    trace_path=None,
    cache_path=None,
) -> str:
    """Полный цикл: шаги 1-5 алгоритма пересказа."""
    profile = PROFILES[detail]

    def note(msg: str) -> None:
        if stage:
            stage.info(msg)

    trace: list[str] = []

    # --- Кэш шагов 1-3 (это самые дорогие вызовы модели) ---------------
    key = _cache_key(segments, detail) if cache_path else ""
    cached = _load_cache(cache_path, key) if cache_path else None

    # --- Шаг 1 ---------------------------------------------------------
    if stage:
        stage.info("шаг 1: первичное сканирование (тема, проблема, структура)")
    if cached and "preview" in cached:
        preview = cached["preview"]
        if stage:
            stage.info("шаг 1: взят из кэша")
    else:
        preview = step1_preview(client, segments)
    trace.append("=== ШАГ 1. ПЕРВИЧНОЕ СКАНИРОВАНИЕ ===\n" + preview)
    if stage:
        stage.info("шаг 1 завершён")

    # --- Шаги 2-3 по блокам -------------------------------------------
    chunks = split_chunks(segments)
    cached_theses = (cached or {}).get("theses") or []
    if stage:
        stage.info(f"шаги 2-3: разбор {len(chunks)} блоков (тезисы + фильтрация)")
    theses_by_chunk: list[tuple[Chunk, list[str]]] = []
    for i, ch in enumerate(chunks):
        if i < len(cached_theses):
            theses = list(cached_theses[i])
            if stage and len(chunks) > 1:
                stage.progress((i + 1) / len(chunks) * 70,
                               f"блок {i + 1}/{len(chunks)}: {len(theses)} тезисов (кэш)")
        else:
            theses = step2_extract(client, ch)
            if stage and len(chunks) > 1:
                stage.progress((i + 1) / len(chunks) * 70,
                               f"блок {i + 1}/{len(chunks)}: {len(theses)} тезисов")
            # сохраняем немедленно: сбой на шаге 4 не должен терять работу
            if cache_path:
                done = [t for _, t in theses_by_chunk] + [theses]
                _save_cache(cache_path, key, {
                    "preview": preview,
                    "theses": done,
                })
        theses_by_chunk.append((ch, theses))
        trace.append(
            f"=== ШАГИ 2-3. БЛОК {i + 1} ({fmt_duration(ch.start)}–{fmt_duration(ch.end)}) ===\n"
            + ("\n".join(f"- {t}" for t in theses) or "(без тезисов)")
        )
    if stage:
        stage.info("шаги 2-3 завершены")

    if cache_path and not cached:
        _save_cache(cache_path, key, {
            "preview": preview,
            "theses": [t for _, t in theses_by_chunk],
        })

    # --- Разделы по хронологии ----------------------------------------
    sections = _themes_with_time(client, segments, preview, profile["sections"])

    # --- Шаг 4 ---------------------------------------------------------
    if stage:
        stage.info("шаг 4: написание черновика своими словами")
    draft = step4_draft(client, preview, theses_by_chunk, detail, sections)
    trace.append("=== ШАГ 4. ЧЕРНОВИК ===\n" + draft)
    if stage:
        stage.info("шаг 4 завершён")

    # --- Шаг 5 ---------------------------------------------------------
    if stage:
        stage.info("шаг 5: редактура и проверка нейтральности")
    final = step5_edit(client, draft)
    trace.append("=== ШАГ 5. ИТОГОВЫЙ ТЕКСТ ===\n" + final)
    if stage:
        stage.info("шаг 5 завершён")

    if trace_path:
        try:
            from pathlib import Path as _P
            _P(trace_path).write_text("\n\n".join(trace), encoding="utf-8")
        except OSError:
            pass

    header = [
        f"РЕФЕРАТ: {title}",
        f"Длительность: {fmt_duration(duration)} | "
        f"Детализация: {detail} ({_level_name(detail)}) | "
        f"Метод: абстрактивный пересказ (5 шагов)",
        "",
    ]
    body = final.strip()

    # дословные цитаты — из самого транскрипта, а не из ответа модели
    if profile["quotes"]:
        quotes = _extract_quotes(segments, profile["quotes"])
        if quotes:
            body += "\n\n## ДОСЛОВНЫЕ ЦИТАТЫ ИЗ МАТЕРИАЛА\n"
            for start, text in quotes:
                body += f'- «{text}» ({fmt_duration(start)})\n'

    return "\n".join(header) + body + "\n"


def _level_name(level: int) -> str:
    return {1: "кратко", 2: "сжато", 3: "нормально",
            4: "подробно", 5: "детально"}.get(level, "")
