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
почти всегда больше. Большие файлы режутся на части модулем audio_chunking
(через imageio-ffmpeg, без перекодирования). Чанки отправляются в Groq
ПАРАЛЛЕЛЬНО (с ограничением на число одновременных запросов), а не по
очереди — это в разы ускоряет обработку длинных визитов: время работы
определяется самым медленным чанком, а не суммой всех.
"""

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from audio_chunking import get_chunk_offset_seconds, needs_chunking, split_audio_into_chunks

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3"

# Groq free tier: 30 запросов в минуту, 7200 секунд аудио в час. Ограничиваем
# параллелизм с запасом от лимита RPM, а не отправляем все чанки сразу.
MAX_PARALLEL_CHUNKS = 5


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


def transcribe_with_chunking(file_bytes: bytes, filename: str, api_key: str) -> str:
    """Транскрибирует аудио любого размера и сразу возвращает читаемый текст
    с тайм-кодами. Если файл укладывается в лимит Groq — отправляет напрямую;
    если больше — режет на части (audio_chunking) и отправляет чанки
    ПАРАЛЛЕЛЬНО (до MAX_PARALLEL_CHUNKS одновременно), затем сшивает
    результат в один текст со сквозными тайм-кодами."""
    if not needs_chunking(file_bytes):
        result = _transcribe_single_file(file_bytes, filename, api_key)
        return build_plain_transcript(result)

    chunks = split_audio_into_chunks(file_bytes, filename)

    # {chunk_index: list_of_segments} — заполняется по мере завершения
    # параллельных запросов, порядок восстанавливаем по индексу в конце.
    results_by_index: dict[int, list[dict]] = {}

    def _process_chunk(index: int, chunk_bytes: bytes) -> tuple[int, list[dict]]:
        chunk_result = _transcribe_single_file(chunk_bytes, f"chunk_{index:03d}.m4a", api_key)
        offset = get_chunk_offset_seconds(index)
        segments = []
        for segment in chunk_result.get("segments", []):
            shifted = dict(segment)
            shifted["start"] = segment.get("start", 0) + offset
            shifted["end"] = segment.get("end", 0) + offset
            segments.append(shifted)
        return index, segments

    errors = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
        futures = [
            executor.submit(_process_chunk, index, chunk_bytes)
            for index, chunk_bytes in enumerate(chunks)
        ]
        for future in as_completed(futures):
            try:
                index, segments = future.result()
                results_by_index[index] = segments
            except GroqTranscriptionError as e:
                errors.append(str(e))

    if errors:
        raise GroqTranscriptionError(
            f"Не удалось распознать {len(errors)} из {len(chunks)} частей аудио: {errors[0]}"
        )

    all_segments = []
    for index in sorted(results_by_index.keys()):
        all_segments.extend(results_by_index[index])

    return build_plain_transcript({"segments": all_segments})


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
