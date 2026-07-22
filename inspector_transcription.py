"""
Транскрибация длинных аудиозаписей (60-90 минут) через DashScope (fun-asr).
Поддерживает диаризацию говорящих и тайм-коды на уровне предложений.

DashScope не принимает файл напрямую в этом асинхронном режиме — только
публичный file_url. Поэтому файл нужно на время отдать по HTTP-ссылке
(см. file_server.py — раздача из оперативной памяти, без записи на диск).
"""

import json
import time
import urllib.error
import urllib.request

DASHSCOPE_ASR_SUBMIT_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/asr/transcription"
DASHSCOPE_TASK_QUERY_URL = "https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"

ASR_MODEL_LONG = "fun-asr"  # поддерживает файлы до 12 часов, диаризацию, тайм-коды


class TranscriptionError(Exception):
    pass


def submit_transcription_task(file_url: str, api_key: str) -> str:
    """Отправляет асинхронную задачу транскрибации в DashScope. Возвращает task_id."""
    payload = json.dumps(
        {
            "model": ASR_MODEL_LONG,
            "input": {"file_urls": [file_url]},
            "parameters": {
                "diarization_enabled": True,
                "timestamp_alignment_enabled": True,
                "language_hints": ["ru"],
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        DASHSCOPE_ASR_SUBMIT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-DashScope-Async": "enable",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        task_id = result.get("output", {}).get("task_id")
        if not task_id:
            raise TranscriptionError(f"Не удалось получить task_id: {result}")
        return task_id
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise TranscriptionError(f"Ошибка при отправке задачи ({e.code}): {body[:300]}")


def poll_transcription_task(task_id: str, api_key: str, timeout_seconds: int = 900, poll_interval: int = 10) -> dict:
    """Опрашивает статус задачи, пока она не завершится (успешно или с ошибкой)."""
    url = DASHSCOPE_TASK_QUERY_URL.format(task_id=task_id)
    req_headers = {"Authorization": f"Bearer {api_key}"}

    elapsed = 0
    while elapsed < timeout_seconds:
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise TranscriptionError(f"Ошибка при опросе задачи ({e.code}): {body[:300]}")

        status = result.get("output", {}).get("task_status")

        if status == "SUCCEEDED":
            return result
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise TranscriptionError(f"Задача транскрибации завершилась со статусом {status}: {result}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TranscriptionError(f"Транскрибация не завершилась за {timeout_seconds} секунд (task_id={task_id})")


def fetch_transcription_result(result_url: str) -> dict:
    """Скачивает итоговый JSON с результатами транскрибации по ссылке из ответа задачи."""
    req = urllib.request.Request(result_url)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_timestamp(ms: int) -> str:
    """Переводит миллисекунды в формат ЧЧ:ММ:СС."""
    total_seconds = ms // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_readable_transcript(transcription_json: dict) -> str:
    """Собирает читаемую расшифровку вида 'Спикер N [ММ:СС]: текст' из ответа DashScope."""
    lines = []
    transcripts = transcription_json.get("transcripts", [])

    for channel in transcripts:
        for sentence in channel.get("sentences", []):
            speaker_id = sentence.get("speaker_id")
            speaker_label = f"Спикер {speaker_id + 1}" if speaker_id is not None else "Спикер"
            timestamp = _format_timestamp(sentence.get("begin_time", 0))
            text = sentence.get("text", "").strip()
            if text:
                lines.append(f"{speaker_label} [{timestamp}]: {text}")

    return "\n".join(lines)


def transcribe_long_audio(file_url: str, api_key: str) -> str:
    """Полный цикл: отправка задачи → ожидание → скачивание результата → читаемый текст."""
    task_id = submit_transcription_task(file_url, api_key)
    task_result = poll_transcription_task(task_id, api_key)

    result_url = None
    results = task_result.get("output", {}).get("results", [])
    if results:
        result_url = results[0].get("transcription_url") or results[0].get("subtask_status", {}).get("url")

    if not result_url:
        raise TranscriptionError(f"В ответе задачи нет ссылки на результат: {task_result}")

    transcription_json = fetch_transcription_result(result_url)
    return build_readable_transcript(transcription_json)
