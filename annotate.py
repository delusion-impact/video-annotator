#!/usr/bin/env python3
"""Поэтапный пайплайн: видео -> аудио -> транскрипт -> текстовая аннотация.

Примеры:
    python annotate.py video.mp4
    python annotate.py video.mp4 --detail 5
    python annotate.py video.mp4 -d 2 -o annotation.txt --save-transcript
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import media, transcribe
from pipeline.summary import DETAIL_PROFILES, build_annotation
from pipeline.ui import Stage, StageError, fmt_duration, parse_detail

TOTAL_STEPS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Обработка видео: транскрибация аудио и текстовая аннотация.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Уровни детализации: 1 (кратко) .. 5 (детально)",
    )
    p.add_argument("video", help="путь к видеофайлу")
    p.add_argument("-d", "--detail", default="3",
                   help="степень детализации аннотации: 1..5 или кратко/подробно/детально (по умолчанию 3)")
    p.add_argument("-o", "--out", metavar="FILE",
                   help="сохранить аннотацию в файл (иначе вывод в консоль)")
    p.add_argument("--save-transcript", metavar="FILE",
                   help="сохранить транскрипт с таймкодами в файл")
    p.add_argument("--lang", default="ru",
                   help="язык речи (по умолчанию ru; можно auto)")
    p.add_argument("--model", default="small",
                   help="модель whisper: tiny/base/small/medium/large-v3 (по умолчанию small)")
    p.add_argument("--threads", type=int, default=None,
                   help="число потоков whisper (по умолчанию до 16)")
    p.add_argument("--whisper-cli", default=None,
                   help="путь к whisper-cli.exe (иначе PATH или tools/whisper)")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                   help="где считать: auto (CUDA если есть), gpu, cpu (по умолчанию auto)")
    p.add_argument("--force", action="store_true",
                   help="пересчитать транскрипт, даже если есть кэш")
    p.add_argument("--keep-wav", action="store_true",
                   help="не удалять промежуточный WAV после обработки")
    p.add_argument("--work-dir", default=None,
                   help="папка для промежуточных файлов (по умолчанию <видео>/../.video-annotator)")
    p.add_argument("--mode", default="referat", choices=["referat", "extract"],
                   help="способ аннотации: referat (пересказ LLM по 5 шагам) "
                        "или extract (локальная выборка предложений)")
    p.add_argument("--llm-model", default=None,
                   help="модель LLM для пересказа (по умолчанию opencode/minimax-m3.1-flash-free)")
    p.add_argument("--opencode-cli", default=None,
                   help="путь к opencode-cli.exe")
    p.add_argument("--save-trace", metavar="FILE",
                   help="сохранить пошаговый разбор (шаги 1-5) в файл")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        detail = parse_detail(args.detail)
    except StageError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"[FAIL] файл не найден: {video}", file=sys.stderr)
        return 2

    work = (Path(args.work_dir).expanduser().resolve() if args.work_dir
            else video.parent / ".video-annotator")
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{video.stem}.wav"
    transcript_json = work / f"{video.stem}.json"

    started = time.time()
    print("=" * 60)
    print(f"Видео: {video.name}")
    print(f"Детализация аннотации: {detail} — {DETAIL_PROFILES[detail]['name']}")
    print("=" * 60)

    # ---- Этап 1: анализ входного файла -------------------------------
    stage = Stage(1, TOTAL_STEPS, "Анализ входного файла")
    try:
        info = media.probe(video)
        stage.ok(
            f"{fmt_duration(info['duration'])}, "
            f"{info['width']}x{info['height']}, "
            f"видео {info['video_codec']}, аудио {info['audio_codec']}"
        )
    except StageError as exc:
        stage.fail(str(exc))
        return 1

    # ---- Этап 2: извлечение аудио ------------------------------------
    stage = Stage(2, TOTAL_STEPS, "Извлечение аудио (ffmpeg, 16 кГц моно)")
    try:
        if wav.exists() and wav.stat().st_size > 0:
            stage.info("используется ранее извлечённое аудио")
        else:
            media.extract_audio(video, wav, info["duration"], stage)
        stage.ok(f"WAV: {wav.name} ({wav.stat().st_size / 1_048_576:.0f} МБ)")
    except StageError as exc:
        stage.fail(str(exc))
        return 1

    # ---- Этап 3: транскрибация ---------------------------------------
    stage = Stage(3, TOTAL_STEPS, "Транскрибация аудио (whisper.cpp)")
    try:
        json_path = transcribe.transcribe(
            wav=wav,
            model=args.model,
            cli_path=args.whisper_cli,
            lang=args.lang,
            threads=args.threads,
            out_json=transcript_json,
            stage=stage,
            force=args.force,
            device=args.device,
            duration=info["duration"],
        )
        segments = transcribe.load_segments(json_path)
        words = sum(len(s["text"].split()) for s in segments)
        stage.ok(f"{len(segments)} сегментов, ~{words:,} слов".replace(",", " "))
    except StageError as exc:
        stage.fail(str(exc))
        return 1

    if args.save_transcript:
        out_tr = Path(args.save_transcript).expanduser().resolve()
        out_tr.parent.mkdir(parents=True, exist_ok=True)
        transcribe.save_transcript_txt(segments, out_tr)
        stage.info(f"транскрипт сохранён: {out_tr}")

    # ---- Этап 4: аннотация -------------------------------------------
    if args.mode == "referat":
        stage = Stage(4, TOTAL_STEPS,
                      "Подготовка реферата (5 шагов: сканирование → маркировка → "
                      "фильтрация → пересказ → редактура)")
    else:
        stage = Stage(4, TOTAL_STEPS, "Создание аннотации (локальная выборка)")
    try:
        if args.mode == "referat":
            from pipeline.llm import DEFAULT_MODEL, LLMClient
            from pipeline.referat import build_llm_annotation

            client = LLMClient(
                model=args.llm_model or DEFAULT_MODEL,
                cli=args.opencode_cli,
            )
            if not client.available:
                raise StageError(
                    "opencode-cli не найден. Укажите путь через --opencode-cli "
                    "или используйте --mode extract"
                )
            stage.info(f"LLM: {client.model}")
            annotation = build_llm_annotation(
                client=client,
                segments=segments,
                detail=detail,
                duration=info["duration"],
                title=video.name,
                stage=stage,
                trace_path=(str(Path(args.save_trace).expanduser().resolve())
                            if args.save_trace else None),
                cache_path=str(work / f"{video.stem}.referat_cache.json"),
            )
        else:
            annotation = build_annotation(
                segments=segments,
                detail=detail,
                duration=info["duration"],
                title=video.name,
            )
        stage.ok(f"{len(annotation.split())} слов в аннотации")
    except (StageError, ValueError) as exc:
        stage.fail(str(exc))
        return 1

    print()
    print("-" * 60)
    print(annotation.rstrip())
    print("-" * 60)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(annotation, encoding="utf-8")
        print(f"Аннотация сохранена: {out_path}")

    if not args.keep_wav and wav.exists():
        try:
            wav.unlink()
        except OSError:
            pass

    total = time.time() - started
    print(f"Готово за {fmt_duration(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
