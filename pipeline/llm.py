"""Клиент LLM через opencode CLI.

Используется opencode-cli.exe с командой `run --format json`: это позволяет
генерировать абстрактивный пересказ (реферат) локально, без внешних API-ключей.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .ui import StageError

DEFAULT_MODEL = "opencode/minimax-m3.1-flash-free"

CLI_CANDIDATES = [
    Path.home() / "AppData/Roaming/ai.opencode.desktop/cli",
    Path.home() / ".config/opencode",
    Path.home() / ".local/share/opencode",
]


def find_cli(explicit: str | None = None) -> str | None:
    """Ищет opencode-cli.exe: явно -> PATH -> стандартные места установки."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        raise StageError(f"opencode-cli не найден: {explicit}")

    on_path = shutil.which("opencode-cli") or shutil.which("opencode")
    if on_path:
        return on_path

    found: list[Path] = []
    for base in CLI_CANDIDATES:
        if base.exists():
            found.extend(base.rglob("opencode-cli.exe"))
    if found:
        # берём самую свежую версию
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(found[0])
    return None


class LLMClient:
    """Обёртка над `opencode run` для генерации текста."""

    def __init__(self, model: str = DEFAULT_MODEL, cli: str | None = None,
                 timeout: int = 600, retries: int = 2) -> None:
        self.model = model
        self.cli = find_cli(cli)
        self.timeout = timeout
        self.retries = retries

    @property
    def available(self) -> bool:
        return self.cli is not None

    def complete(self, prompt: str, session: str | None = None) -> str:
        """Отправляет промпт и возвращает текст ответа."""
        if not self.cli:
            raise StageError("opencode-cli не найден — абстрактивный пересказ недоступен")

        cmd = [self.cli, "run", "--model", self.model, "--format", "json"]
        if session:
            cmd += ["--session", session]
        cmd.append(prompt)

        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                last_error = "таймаут ответа модели"
                time.sleep(2)
                continue

            if proc.returncode != 0:
                last_error = (proc.stderr or proc.stdout or "").strip()[:300]
                time.sleep(2)
                continue

            text = self._extract_text(proc.stdout)
            if text.strip():
                return text.strip()

            last_error = "пустой ответ модели"
            time.sleep(2)

        raise StageError(f"не удалось получить ответ от модели: {last_error}")

    @staticmethod
    def _extract_text(stream: str) -> str:
        """Достаёт текстовые части из JSONL-вывода opencode."""
        parts: list[str] = []
        for line in stream.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part") or {}
            if event.get("type") == "text" and part.get("type") == "text":
                text = part.get("text")
                if text:
                    parts.append(text)
            elif part.get("type") == "text" and part.get("text"):
                parts.append(part["text"])
        return "\n".join(parts)
