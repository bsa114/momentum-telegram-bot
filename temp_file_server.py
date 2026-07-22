"""
Временное файловое хранилище в оперативной памяти — чтобы отдать
DashScope прямую HTTPS-ссылку на аудиофайл без записи на диск.

Каждый файл живёт под случайным id и удаляется сразу после того,
как DashScope его скачал (либо по таймауту, на всякий случай).
"""

import asyncio
import time
import uuid

from aiohttp import web

# file_id -> {"bytes": ..., "content_type": ..., "created_at": ...}
_file_store: dict[str, dict] = {}

FILE_TTL_SECONDS = 60 * 30  # 30 минут на всякий случай, если что-то пошло не так


def store_file_in_memory(file_bytes: bytes, content_type: str = "audio/ogg") -> str:
    """Кладёт файл в память и возвращает его временный id."""
    file_id = uuid.uuid4().hex
    _file_store[file_id] = {
        "bytes": file_bytes,
        "content_type": content_type,
        "created_at": time.time(),
    }
    return file_id


def pop_file_from_memory(file_id: str) -> None:
    """Удаляет файл из памяти (вызывается после успешной транскрибации)."""
    _file_store.pop(file_id, None)


def _cleanup_expired_files() -> None:
    now = time.time()
    expired_ids = [
        fid for fid, data in _file_store.items()
        if now - data["created_at"] > FILE_TTL_SECONDS
    ]
    for fid in expired_ids:
        _file_store.pop(fid, None)


async def serve_temp_file(request: web.Request) -> web.Response:
    """Endpoint /tmp-audio/{file_id} — отдаёт файл по id, если он ещё в памяти."""
    _cleanup_expired_files()
    file_id = request.match_info.get("file_id", "")
    data = _file_store.get(file_id)
    if not data:
        return web.Response(status=404, text="Файл не найден или уже удалён")
    return web.Response(body=data["bytes"], content_type=data["content_type"])


async def periodic_cleanup_task() -> None:
    """Фоновая задача — периодически чистит зависшие файлы."""
    while True:
        await asyncio.sleep(300)
        _cleanup_expired_files()
