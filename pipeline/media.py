"""Работа с ffmpeg/ffprobe: анализ видео и извлечение аудио."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .ui import Stage, StageError, fmt_duration

PROGRESS_KEYS = ("out_time_us", "out_time_ms", "out_time")


def find_tool(name: str) -> str:
    """Находит ffmpeg/ffprobe в PATH или в пакете winget."""
    path = shutil.which(name)
    if path:
        return path
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    hits = sorted(packages.glob(f"Gyan.FFmpeg_*/*/{name}.exe")) or sorted(
        packages.glob(f"Gyan.FFmpeg_*/{name}.exe")
    )
    if hits:
        return str(hits[0])
    raise StageError(
        f"{name} не найден. Установите: winget install Gyan.FFmpeg"
    )


def probe(video: Path) -> dict:
    """Возвращает параметры файла: длительность, потоки, разрешение."""
    ffprobe = find_tool("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise StageError(f"ffprobe не смог прочитать файл: {proc.stderr.strip()[:300]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise StageError(f"некорректный ответ ffprobe: {exc}") from exc

    streams = data.get("streams", [])
    video_s = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_s = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video_s is None:
        raise StageError("в файле нет видеопотока")
    if audio_s is None:
        raise StageError("в файле нет аудиодорожки — транскрибация невозможна")

    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise StageError("не удалось определить длительность видео")

    return {
        "duration": duration,
        "size": int(data.get("format", {}).get("size") or 0),
        "width": video_s.get("width"),
        "height": video_s.get("height"),
        "video_codec": video_s.get("codec_name"),
        "audio_codec": audio_s.get("codec_name"),
        "sample_rate": int(audio_s.get("sample_rate") or 0),
    }


def extract_audio(video: Path, wav: Path, duration: float, stage: Stage) -> None:
    """Декодирует аудио в WAV 16 кГц моно (формат для whisper)."""
    ffmpeg = find_tool("ffmpeg")
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-progress", "pipe:1", str(wav),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    errors: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if "=" not in line:
            if line:
                errors.append(line)
            continue
        key, _, value = line.partition("=")
        if key in PROGRESS_KEYS:
            try:
                seconds = float(value) / 1_000_000
            except ValueError:
                continue
            if duration > 0:
                stage.progress(seconds / duration * 100, fmt_duration(seconds))
        elif key == "progress" and value == "end":
            stage.progress(100)
        elif key == "progress":
            pass
        else:
            errors.append(line)
    code = proc.wait()
    if code != 0:
        tail = " ".join(errors[-3:])[:300]
        raise StageError(f"ffmpeg завершился с кодом {code}: {tail}")
    if not wav.exists() or wav.stat().st_size == 0:
        raise StageError("ffmpeg не создал аудиофайл")


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def cache_valid(meta_path: Path, expected: dict) -> bool:
    meta = load_json(meta_path)
    return meta is not None and all(meta.get(k) == v for k, v in expected.items())
