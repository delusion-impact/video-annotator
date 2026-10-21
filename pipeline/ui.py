"""Поэтапный вывод статусов пайплайна."""

from __future__ import annotations

import sys
import time


class StageError(Exception):
    """Ошибка текущего этапа (сообщение выводится как [FAIL])."""


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def plural_words(n: int) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        w = "слово"
    elif 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        w = "слова"
    else:
        w = "слов"
    return f"{n:,}".replace(",", " ") + " " + w


class Stage:
    """Статус одного этапа: заголовок, прогресс, итог [OK]/[FAIL]."""

    def __init__(self, idx: int, total: int, title: str):
        self.idx = idx
        self.total = total
        self.title = title
        self.t0 = time.time()
        self._pct: int | None = None
        print(f"[{idx}/{total}] {title}", flush=True)

    def info(self, msg: str) -> None:
        self.clear_progress()
        print(f"       {msg}", flush=True)

    def progress(self, pct: float, note: str = "") -> None:
        pct = max(0, min(100, int(pct)))
        if pct == self._pct:
            return
        self._pct = pct
        filled = pct // 5
        bar = "#" * filled + "-" * (20 - filled)
        eta = ""
        if 0 < pct < 100:
            elapsed = time.time() - self.t0
            remain = elapsed * (100 - pct) / pct
            if remain >= 90:
                eta = f" | осталось ~{remain / 60:.0f} мин"
            elif remain >= 2:
                eta = f" | осталось ~{remain:.0f} с"
        note_s = f" {note}" if note else ""
        sys.stdout.write(f"\r       [{bar}] {pct:3d}%{note_s}{eta}   ")
        sys.stdout.flush()

    def clear_progress(self) -> None:
        if self._pct is not None:
            sys.stdout.write("\r" + " " * 79 + "\r")
            sys.stdout.flush()
            self._pct = None

    def ok(self, msg: str = "") -> None:
        self.clear_progress()
        dt = time.time() - self.t0
        parts = [p for p in (msg, f"{dt:.1f} с") if p]
        print(f"       [OK] {' | '.join(parts)}", flush=True)

    def fail(self, msg: str) -> None:
        self.clear_progress()
        print(f"       [FAIL] {msg}", flush=True)


def parse_detail(value: str) -> int:
    """Преобразует --detail в уровень 1..5 (поддерживает псевдонимы)."""
    aliases = {
        "1": 1, "brief": 1, "кратко": 1, "мин": 1, "минимально": 1,
        "2": 2, "short": 2, "сжато": 2, "средне-кратко": 2,
        "3": 3, "normal": 3, "нормально": 3, "средне": 3, "обычно": 3,
        "4": 4, "long": 4, "подробно": 4, "развернуто": 4,
        "5": 5, "detailed": 5, "детально": 5, "макс": 5, "максимально": 5,
    }
    level = aliases.get(value.strip().lower())
    if level is None:
        raise StageError(
            f"неизвестный уровень детализации: {value!r} "
            "(допустимо: 1..5, кратко/сжато/нормально/подробно/детально "
            "или brief/short/normal/long/detailed)"
        )
    return level
