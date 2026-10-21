#!/usr/bin/env python3
"""Самопроверка пайплайна: запускается без видео и внешних сервисов.

    python selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.summary import DETAIL_PROFILES, build_annotation
from pipeline.ui import fmt_duration, parse_detail

FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "OK  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def make_segments(n: int = 120) -> list[dict]:
    """Синтетический транскрипт для проверки суммаризации."""
    topics = [
        "ребенок постепенно привыкает к режим�� кормления и спокойно засыпает",
        "мама рассказывает про дневной сон и про особенности режима",
        "когда ребенок просыпается ночью, важно дать ему грудь и успокоить",
        "поддержание внутренних часов помогает ребенку лучше ориентироваться",
        "постепенное отлучение от груди требует терпения и мягкости",
    ]
    segments = []
    for i in range(n):
        text = topics[i % len(topics)]
        segments.append({
            "start": i * 6.0,
            "end": i * 6.0 + 6.0,
            "text": text + ".",
        })
    return segments


def main() -> int:
    print("=" * 60)
    print("Самопроверка video-annotator")
    print("=" * 60)

    # 1. Уровни детализации
    for raw, expected in [("1", 1), ("5", 5), ("кратко", 1), ("детально", 5),
                          ("normal", 3), ("подробно", 4)]:
        try:
            got = parse_detail(raw)
            check(f"парсинг уровня '{raw}'", got == expected, f"-> {got}")
        except Exception as exc:  # noqa: BLE001
            check(f"парсинг уровня '{raw}'", False, str(exc))
    try:
        parse_detail("999")
        check("некорректный уровень отклоняется", False)
    except Exception:  # noqa: BLE001
        check("некорректный уровень отклоняется", True)

    # 2. Формат длительности
    check("формат времени", fmt_duration(7841.68) == "2:10:42", fmt_duration(7841.68))
    check("формат времени (минуты)", fmt_duration(125) == "2:05", fmt_duration(125))

    # 3. Аннотация на всех уровнях
    segments = make_segments()
    duration = 120 * 6.0
    prev_words = 0
    for level in range(1, 6):
        ann = build_annotation(segments, level, duration, "test.mp4")
        words = len(ann.split())
        check(f"уровень {level} формируется", len(ann) > 100, f"{words} слов")
        check(f"уровень {level} не пустеет", words > prev_words,
              f"{words} > {prev_words}")
        for section in ("КРАТКОЕ СОДЕРЖАНИЕ", "КЛЮЧЕВЫЕ СЛОВА"):
            check(f"уровень {level}: есть '{section}'", section in ann)
        if level >= 2:
            check(f"уровень {level}: есть тезисы", "ОСНОВНЫЕ ТЕЗИСЫ" in ann)
        if level >= 3:
            check(f"уровень {level}: есть хронология", "РАЗДЕЛЫ ПО ХРОНОЛОГИИ" in ann)
        if level >= 4:
            check(f"уровень {level}: есть цитаты", "ЦИТАТЫ" in ann)
        prev_words = words

    # 4. Профили детализации согласованы
    check("профили 1..5 определены",
          all(l in DETAIL_PROFILES for l in range(1, 6)))
    check("объём растёт с уровнем",
          DETAIL_PROFILES[5]["words"] > DETAIL_PROFILES[1]["words"])

    # 5. Защита от пустого транскрипта
    try:
        build_annotation([], 3, 100.0, "empty.mp4")
        check("пустой транскрипт обрабатывается", False, "должен бросать ValueError")
    except ValueError:
        check("пустой транскрипт обрабатывается", True)

    # 6. Разбор JSON whisper
    from pipeline import transcribe
    with tempfile.TemporaryDirectory() as tmp:
        jf = Path(tmp) / "t.json"
        jf.write_text(json.dumps({
            "transcription": [
                {"offsets": {"from": 0, "to": 1500}, "text": " Привет."},
                {"offsets": {"from": 1500, "to": 3000}, "text": " Как дела?"},
                {"offsets": {"from": 3000, "to": 4000}, "text": "   "},
            ]
        }, ensure_ascii=False), encoding="utf-8")
        segs = transcribe.load_segments(jf)
        check("JSON разбирается", len(segs) == 2, f"{len(segs)} сегмента")
        check("таймкоды в секундах", abs(segs[1]["start"] - 1.5) < 1e-6,
              f"start={segs[1]['start']}")
        check("пустые сегменты отбрасываются", all(s["text"] for s in segs))

    # 7. Рефератный модуль: разбивка на блоки и чистка текста
    from pipeline import referat
    segs = make_segments(400)
    chunks = referat.split_chunks(segs, chunk_words=200)
    check("транскрипт режется на блоки", len(chunks) > 1, f"{len(chunks)} блоков")
    check("блоки идут по порядку",
          all(chunks[i].start <= chunks[i + 1].start for i in range(len(chunks) - 1)))
    check("блоки не теряют текст",
          sum(len(c.text.split()) for c in chunks) >= len(segs) * 0.9)
    check("чистка убирает слова-паразиты",
          "как бы" not in referat._clean("Ну вот, как бы сказать, это факт"))
    check("профили реферата определены",
          all(l in referat.PROFILES for l in range(1, 6)))
    check("объём реферата растёт с уровнем",
          referat.PROFILES[5]["words"] > referat.PROFILES[1]["words"])
    check("число аргументов растёт с уровнем",
          referat.PROFILES[5]["args"] > referat.PROFILES[1]["args"])

    # 8. Поиск opencode-cli (не требует запуска модели)
    from pipeline.llm import find_cli
    cli = find_cli()
    check("opencode-cli найден", cli is not None, cli or "не найден")

    # 9. Разбор тезисов: модель может оформлять их по-разному
    raw = (
        "[Первый тезис про массаж и тонус ребенка]\n"
        "- Второй тезис через дефис\n"
        "* Третий тезис через звездочку\n"
        "1. Четвертый тезис через нумерацию\n"
        "Совсем коротко\n"
    )
    parsed = referat._parse_theses(raw)
    check("тезисы разбираются в скобках", len(parsed) == 4, f"{len(parsed)} шт.")
    check("тезисы разбираются с дефисом", any(t.startswith("Второй") for t in parsed))
    check("тезисы разбираются с нумерацией",
          any(t.startswith("Четвертый") for t in parsed))
    check("короткие строки отбрасываются", not any(t == "Совсем коротко" for t in parsed))
    check("пустой ответ даёт пустой список", referat._parse_theses("НЕТ ТЕЗИСОВ") == [])

    # 10. Кэш шагов 1-3
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = Path(tmp) / "cache.json"
        key = referat._cache_key(segs, 3)
        check("ключ кэша стабилен", key == referat._cache_key(segs, 3))
        check("ключ кэша зависит от уровня", key != referat._cache_key(segs, 5))
        referat._save_cache(str(cache_file), key,
                            {"preview": "ТЕМА: x", "theses": [["тезис"]]})
        loaded = referat._load_cache(str(cache_file), key)
        check("кэш сохраняется и читается", loaded is not None and "preview" in loaded)
        check("кэш отбрасывается при чужом ключе",
              referat._load_cache(str(cache_file), "другой") is None)
        check("пустой кэш не ломает чтение",
              referat._load_cache(str(Path(tmp) / "нет.json"), key) is None)

    # 11. Промпт шага 4 собирается без ошибок (регрессия: extra.append на строке)
    from pipeline.referat import STEP4_PROMPT
    try:
        prompt = STEP4_PROMPT.format(
            preview="ТЕМА: x", theses="- тезис", n_args=5, words=500,
            extra="разделы\n- 0:00 тема\n",
        )
        check("промпт шага 4 собирается", "тезис" in prompt and "500" in prompt)
    except (IndexError, AttributeError, KeyError) as exc:
        check("промпт шага 4 собирается", False, f"{type(exc).__name__}: {exc}")

    # 12. Извлечение дословных цитат из транскрипта
    quotes = referat._extract_quotes(segs, 2)
    check("цитаты извлекаются", len(quotes) == 2, f"{len(quotes)} шт.")
    check("цитаты содержат таймкод", all(isinstance(t, float) for t, _ in quotes))
    check("цитаты непустые", all(len(txt) > 20 for _, txt in quotes))

    print("=" * 60)
    if FAILED:
        print(f"ПРОВАЛЕНО проверок: {len(FAILED)}")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("Все проверки пройдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
