"""
Приём больших аудиофайлов (>20 МБ) через ссылку вместо самого файла —
Telegram Bot API не даёт скачать файлы крупнее 20 МБ через getFile,
поэтому для длинных записей визита (60-90 минут) единственный надёжный
путь — принять от пользователя ссылку на файл, размещённый в облаке.

Поддерживаются:
- Яндекс.Диск (публичная ссылка вида https://yadi.sk/d/... или https://disk.yandex.ru/...)
  — через официальный публичный API, без OAuth-токена, без "confirm"-хитростей.
- Google Drive (ссылка вида https://drive.google.com/file/d/<id>/view)
  — с обработкой страницы "не удалось проверить на вирусы" для больших файлов.
- Любая другая прямая ссылка на файл — используется как есть.
"""

import re
import urllib.error
import urllib.parse
import urllib.request
import json

YANDEX_DISK_PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 МБ — разумный потолок для аудио 60-90 минут


class LinkDownloadError(Exception):
    pass


def _is_yandex_disk_link(url: str) -> bool:
    return "yadi.sk" in url or "disk.yandex" in url


def _is_google_drive_link(url: str) -> bool:
    return "drive.google.com" in url


def _resolve_yandex_disk_url(public_url: str) -> str:
    """Получает реальную ссылку на скачивание через официальный публичный API Яндекс.Диска."""
    query = urllib.parse.urlencode({"public_key": public_url})
    request_url = f"{YANDEX_DISK_PUBLIC_API}?{query}"

    req = urllib.request.Request(request_url)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise LinkDownloadError(f"Яндекс.Диск не отдал ссылку на файл ({e.code}): {body[:200]}")

    href = result.get("href")
    if not href:
        raise LinkDownloadError(f"В ответе Яндекс.Диска нет ссылки на скачивание: {result}")
    return href


def _extract_google_drive_file_id(url: str) -> str | None:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None


def _resolve_google_drive_url(url: str) -> str:
    """Строит прямую ссылку на скачивание Google Drive. Для больших файлов
    Google Drive сначала отдаёт HTML-страницу предупреждения о вирусах —
    в реальном скачивании (_download_from_url) это обрабатывается отдельно."""
    file_id = _extract_google_drive_file_id(url)
    if not file_id:
        raise LinkDownloadError("Не удалось распознать id файла в ссылке Google Drive")
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def resolve_direct_download_url(user_provided_url: str) -> str:
    """Определяет тип ссылки и возвращает URL, с которого реально можно скачать байты."""
    url = user_provided_url.strip()

    if _is_yandex_disk_link(url):
        return _resolve_yandex_disk_url(url)
    if _is_google_drive_link(url):
        return _resolve_google_drive_url(url)

    return url  # считаем, что это уже прямая ссылка на файл


def _looks_like_html(content_start: bytes) -> bool:
    lowered = content_start[:500].lower()
    return b"<html" in lowered or b"<!doctype html" in lowered


def download_file_from_link(user_provided_url: str) -> bytes:
    """Скачивает файл по ссылке любого поддерживаемого типа. Возвращает байты.

    Для Google Drive отдельно обрабатывает случай, когда вместо файла
    приходит HTML-страница предупреждения ("не удалось проверить на вирусы") —
    в этом случае достаёт из неё настоящий confirm-токен и повторяет запрос."""
    direct_url = resolve_direct_download_url(user_provided_url)

    req = urllib.request.Request(direct_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read(MAX_DOWNLOAD_BYTES)
    except urllib.error.HTTPError as e:
        raise LinkDownloadError(f"Не удалось скачать файл по ссылке ({e.code})")
    except urllib.error.URLError as e:
        raise LinkDownloadError(f"Не удалось скачать файл по ссылке: {e.reason}")

    if _is_google_drive_link(user_provided_url) and _looks_like_html(content):
        # Большой файл — Google Drive прислал страницу предупреждения вместо файла.
        # Достаём confirm-токен и повторяем запрос уже с ним.
        html_text = content.decode("utf-8", errors="ignore")
        match = re.search(r'confirm=([0-9A-Za-z_-]+)', html_text)
        if not match:
            raise LinkDownloadError(
                "Google Drive вернул страницу подтверждения вместо файла, "
                "но не удалось найти токен подтверждения. Попробуй ссылку на Яндекс.Диск."
            )
        confirm_token = match.group(1)
        file_id = _extract_google_drive_file_id(user_provided_url)
        retry_url = (
            f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
        )
        req = urllib.request.Request(retry_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                content = response.read(MAX_DOWNLOAD_BYTES)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            raise LinkDownloadError(f"Не удалось скачать файл после подтверждения: {e}")

    if len(content) == 0:
        raise LinkDownloadError("Скачанный файл пустой")

    return content
