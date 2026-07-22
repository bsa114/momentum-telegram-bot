"""
Разбивает длинные аудиофайлы на части (чанки) перед отправкой в Groq Whisper —
у Groq жёсткий лимит 25 МБ на файл через прямую загрузку, а полноценная
запись визита 60-90 минут почти всегда его превышает.

Использует imageio-ffmpeg — pip-пакет со встроенным статическим бинарником
ffmpeg, не требующий системной установки (важно для Render, где нет
привилегированного доступа для apt-get). Разбиение идёт с '-c copy'
(без перекодирования) — быстро и без потери качества.
"""

import os
import subprocess
import tempfile

import imageio_ffmpeg

CHUNK_DURATION_SECONDS = 600  # 10 минут — с запасом укладывается в лимит 25 МБ
MAX_SAFE_FILE_SIZE_BYTES = 24 * 1024 * 1024  # чуть меньше лимита Groq (25 МБ)


class AudioChunkingError(Exception):
    pass


def needs_chunking(file_bytes: bytes) -> bool:
    return len(file_bytes) > MAX_SAFE_FILE_SIZE_BYTES


def split_audio_into_chunks(file_bytes: bytes, filename_hint: str = "audio.m4a") -> list[bytes]:
    """Разбивает аудио на части по CHUNK_DURATION_SECONDS секунд.
    Возвращает список чанков в виде байтов, в правильном порядке."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = os.path.splitext(filename_hint)[1] or ".m4a"

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, f"input{suffix}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        output_pattern = os.path.join(tmp_dir, f"chunk_%03d{suffix}")

        result = subprocess.run(
            [
                ffmpeg_path,
                "-i", input_path,
                "-f", "segment",
                "-segment_time", str(CHUNK_DURATION_SECONDS),
                "-c", "copy",
                "-reset_timestamps", "0",
                output_pattern,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise AudioChunkingError(f"ffmpeg завершился с ошибкой: {result.stderr[-500:]}")

        chunk_files = sorted(
            f for f in os.listdir(tmp_dir) if f.startswith("chunk_") and f.endswith(suffix)
        )

        if not chunk_files:
            raise AudioChunkingError("ffmpeg не создал ни одного файла-чанка")

        chunks = []
        for chunk_file in chunk_files:
            with open(os.path.join(tmp_dir, chunk_file), "rb") as f:
                chunks.append(f.read())

        return chunks


def get_chunk_offset_seconds(chunk_index: int) -> int:
    """Смещение по времени (в секундах) для чанка с данным индексом (с нуля) —
    нужно, чтобы правильно сдвинуть тайм-коды при склейке результатов."""
    return chunk_index * CHUNK_DURATION_SECONDS
