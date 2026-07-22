"""
Транскрибация аудио через Groq Whisper (whisper-large-v3) — заменяет
DashScope (fun-asr) для режима Инспектора. По результатам прямого сравнения
на реальном шумном визите (фоновая музыка, посторонние разговоры, несколько
говорящих) Groq Whisper даёт заметно более связный и точный текст.

Ограничение: Groq Whisper НЕ поддерживает диаризацию (разделение по
говорящим) нативно — этим занимается отдельный модуль speaker_diarization.py
через LLM-постобработку уже готового текста.

Groq доступен из России как страны — сервис заблокирован (GroqCloud FAQ:
Greater China, Russia, Syria, Iran, North Korea, Cuba), но наш бот работает
на Render (США/Европа), так что сетевой доступ не блокируется. Проверено
прямым тестовым вызовом с реальным ключом — запрос отработал успешно.

Groq также ограничивает размер одного файла 25 МБ — визит 60-90 минут
почти всегда больше. Поэтому большие файлы автоматически режутся на части
через ffmpeg (предустановлен на Render) перед отправкой, а результаты
сшиваются обратно с сохранением сквозных тайм-кодов.
"""

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Берём с заметным запасом от реального лимита Groq (25 МБ) — конвертация в
# другой формат/битрейт может слегка менять итоговый размер чанка.
MAX_CHUNK_BYTES = 20 * 1024 * 1024
CHUNK_DURATION_SECONDS = 600  # 10 минут — с запасом укладывается в лимит при обычном битрейте голоса


class GroqTranscriptionError(Exception):
    pass


def _format_timestamp(seconds: float) -> str:
    """Переводит секунды (float) в формат ЧЧ:ММ:СС или ММ:СС."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _get_audio_duration_seconds(file_path: str) -> float:
    """Определяет длительность аудио в секундах через ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise GroqTranscriptionError(f"Не удалось определить длительность аудио: {result.stderr}")
    return float(result.stdout.strip())


def _split_audio_into_chunks(file_bytes: bytes, original_filename: str) -> list[tuple[str, float]]:
    """Режет аудио на части по CHUNK_DURATION_SECONDS через ffmpeg, перекодируя
    в компактный mono-opus (небольшой размер при хорошем качестве речи).
    Возвращает список (путь_к_файлу_чанка, время_начала_чанка_в_секундах).
    Файлы чанков остаются на диске после выхода из функции — вызывающий код
    должен удалить их сам после использования."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, f"input_{original_filename}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        duration = _get_audio_duration_seconds(input_path)

        chunk_paths = []
        start = 0.0
        chunk_index = 0
        while start < duration:
            chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_index:03d}.ogg")
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_path,
                    "-ss", str(start), "-t", str(CHUNK_DURATION_SECONDS),
                    "-ac", "1", "-c:a", "libopus", "-b:a", "32k",
                    chunk_path,
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise GroqTranscriptionError(f"Ошибка ffmpeg при нарезке аудио: {result.stderr[-500:]}")

            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()

            # Копируем байты чанка во временный файл, который переживёт выход
            # из этого TemporaryDirectory (нужен вызывающему коду для отправки).
            persistent_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg").name
            with open(persistent_path, "wb") as f:
                f.write(chunk_bytes)

            chunk_paths.append((persistent_path, start))
            start += CHUNK_DURATION_SECONDS
            chunk_index += 1

        return chunk_paths


def _build_multipart_body(file_bytes: bytes, filename: str, boundary: str) -> bytes:
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += file_bytes
    body += f"\r\n--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    body += GROQ_WHISPER_MODEL.encode()
    body += f"\r\n--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="response_format"\r\n\r\n'
    body += b"verbose_json"
    body += f"\r\n--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="language"\r\n\r\n'
    body += b"ru"
    body += f"\r\n--{boundary}--\r\n".encode()
    return bytes(body)


def _transcribe_single_file(file_bytes: bytes, filename: str, api_key: str) -> dict:
    """Отправляет один файл (уже гарантированно меньше лимита) в Groq Whisper."""
    boundary = "----GroqBoundary1234567890"
    body = _build_multipart_body(file_bytes, filename, boundary)

    req = urllib.request.Request(
        GROQ_TRANSCRIPTION_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise GroqTranscriptionError(f"Ошибка Groq Whisper ({e.code}): {body_text[:300]}")
    except urllib.error.URLError as e:
        raise GroqTranscriptionError(f"Не удалось связаться с Groq: {e.reason}")


def transcribe_audio_bytes(file_bytes: bytes, filename: str, api_key: str) -> dict:
    """Транскрибирует аудио любого размера: если файл укладывается в лимит
    Groq — отправляет напрямую; если больше — режет на части через ffmpeg,
    транскрибирует каждую часть отдельно и сшивает результат в единый набор
    сегментов со сквозными тайм-кодами (как будто это был один файл)."""
    if len(file_bytes) <= MAX_CHUNK_BYTES:
        return _transcribe_single_file(file_bytes, filename, api_key)

    chunk_paths = _split_audio_into_chunks(file_bytes, filename)
    all_segments = []

    try:
        for chunk_path, chunk_start_offset in chunk_paths:
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()

            chunk_result = _transcribe_single_file(chunk_bytes, os.path.basename(chunk_path), api_key)

            for segment in chunk_result.get("segments", []):
                shifted_segment = dict(segment)
                shifted_segment["start"] = segment.get("start", 0) + chunk_start_offset
                shifted_segment["end"] = segment.get("end", 0) + chunk_start_offset
                all_segments.append(shifted_segment)
    finally:
        for chunk_path, _ in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return {"segments": all_segments}


def build_plain_transcript(groq_result: dict) -> str:
    """Собирает читаемую расшифровку с тайм-кодами из ответа Groq (без
    разделения по говорящим — это делает отдельный шаг диаризации)."""
    segments = groq_result.get("segments", [])
    lines = []
    for segment in segments:
        timestamp = _format_timestamp(segment.get("start", 0))
        text = segment.get("text", "").strip()
        if text:
            lines.append(f"[{timestamp}] {text}")
    return "\n".join(lines)
