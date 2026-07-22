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
"""

import json
import urllib.error
import urllib.request

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3"


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


def transcribe_audio_bytes(file_bytes: bytes, filename: str, api_key: str) -> dict:
    """Отправляет аудио в Groq Whisper, возвращает результат с сегментами
    и тайм-кодами (verbose_json). Файл передаётся напрямую (multipart),
    без необходимости публичного URL — в отличие от DashScope, здесь не
    нужен промежуточный HTTP-сервер для раздачи из памяти."""
    boundary = "----GroqBoundary1234567890"

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

    req = urllib.request.Request(
        GROQ_TRANSCRIPTION_URL,
        data=bytes(body),
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
