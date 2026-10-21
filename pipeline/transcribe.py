"""Транскрибация через whisper.cpp (whisper-cli)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .ui import Stage, StageError, fmt_duration

# прогресс whisper-cli выводится как "progress = 42%"
_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+(?:\.\d+)?)")

DEFAULT_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"
)
MODEL_NAMES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
}


def find_whisper_cli(explicit: str | None = None, device: str = "auto") -> str:
    """Ищет whisper-cli: указанный путь -> PATH -> локальная сборка.

    device: "auto" — CUDA-сборка, если она есть, иначе CPU;
            "cpu"   — только CPU-сборка;
            "gpu"   — только CUDA-сборка (ошибка, если её нет).
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        raise StageError(f"whisper-cli не найден: {explicit}")

    root = Path(__file__).resolve().parent.parent / "tools"
    cuda_cli = root / "whisper-cuda" / "Release" / "whisper-cli.exe"
    cpu_cli = root / "whisper" / "Release" / "whisper-cli.exe"

    if device == "gpu":
        if cuda_cli.exists():
            return str(cuda_cli)
        raise StageError(
            "CUDA-сборка whisper-cli не найдена. Скачайте whisper-bin-win-cuda-*.zip "
            "в tools/whisper-cuda/"
        )
    if device == "auto" and cuda_cli.exists():
        return str(cuda_cli)

    on_path = shutil.which("whisper-cli") or shutil.which("whisper-cli.exe")
    if on_path:
        return on_path
    if cpu_cli.exists():
        return str(cpu_cli)
    raise StageError(
        "whisper-cli не найден. Скачайте релиз whisper.cpp "
        "(whisper-bin-x64.zip) в папку tools/whisper"
    )


def resolve_model(spec: str) -> str:
    """Путь к .bin либо короткое имя модели (small, base, ...)."""
    p = Path(spec)
    if p.suffix == ".bin":
        if p.exists():
            return str(p)
        raise StageError(f"модель не найдена: {spec}")
    name = MODEL_NAMES.get(spec)
    if name is None:
        raise StageError(f"неизвестная модель {spec!r}; допустимо: {', '.join(MODEL_NAMES)}")
    local = Path(__file__).resolve().parent.parent / "models" / name
    if local.exists():
        return str(local)
    raise StageError(
        f"модель {spec} не скачана: {local}. "
        f"Скачайте: curl -L -o \"{local}\" \"{DEFAULT_MODEL_URL.format(name=name)}\""
    )


def transcribe(
    wav: Path,
    model: str,
    cli_path: str | None,
    lang: str,
    threads: int | None,
    out_json: Path,
    stage: Stage,
    force: bool = False,
    device: str = "auto",
    duration: float = 0.0,
) -> Path:
    """Запускает распознавание и возвращает путь к JSON с сегментами."""
    import os

    cli = find_whisper_cli(cli_path, device=device)
    model_path = resolve_model(model)
    threads = threads or min(16, os.cpu_count() or 4)

    cmd = [
        cli,
        "-m", model_path,
        "-f", str(wav),
        "-l", lang,
        "-t", str(threads),
        "-pp",                      # выводить прогресс (без флага молчит!)
        "-oj",                      # JSON с таймкодами
        "-of", str(out_json.with_suffix("")),
    ]
    if out_json.exists() and not force:
        stage.info(f"кэш транскрипта: {out_json.name}")
        return out_json
    if force and out_json.exists():
        out_json.unlink()
        stage.info("кэш транскрипта сброшен (--force)")

    backend = "GPU (CUDA)" if "whisper-cuda" in cli else "CPU"
    stage.info(f"модель {Path(model_path).name}, {backend}, потоков: {threads}, язык: {lang}")
    if duration:
        # измеренные скорости (секунд аудио в секунду реального времени)
        model_size_gb = Path(model_path).stat().st_size / (1024 ** 3)
        if backend.startswith("GPU"):
            rate = 31.0 if model_size_gb < 1.5 else 13.0
        else:
            rate = 10.0 if model_size_gb < 1.5 else 3.0
        stage.info(f"оценка времени: ~{fmt_duration(duration / rate)} "
                   f"(скорость ~x{rate:.0f} от реального времени)")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    warnings: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        m = _PROGRESS_RE.search(line)
        if m:
            stage.progress(float(m.group(1)))
        elif line and ("error" in line.lower() or "failed" in line.lower()):
            warnings.append(line)
        # остальной вывод (фрагменты текста) игнорируем — он пойдёт в JSON
    code = proc.wait()
    if code != 0:
        tail = " ".join(warnings[-3:])[:300]
        raise StageError(f"whisper-cli завершился с кодом {code}: {tail}")

    if not out_json.exists():
        # часть сборок добавляет .json к -of, часть пишет рядом с wav
        alt = out_json.with_suffix(".json")
        if alt.exists():
            return alt
        raise StageError("whisper-cli не создал JSON с транскриптом")
    return out_json


def load_segments(json_path: Path) -> list[dict]:
    """Читает JSON whisper: [{start, end, text}, ...] в секундах."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"не удалось прочитать транскрипт {json_path}: {exc}") from exc

    segments: list[dict] = []
    for item in data.get("transcription", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        raise StageError("транскрипт пуст: речь в видео не обнаружена")
    return segments


def save_transcript_txt(segments: list[dict], path: Path) -> None:
    lines = []
    for s in segments:
        lines.append(f"[{fmt_duration(s['start'])}] {s['text']}")
    path.write_text("\n".join(lines), encoding="utf-8")
