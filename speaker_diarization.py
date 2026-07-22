"""
Groq Whisper не даёт диаризацию (разделение по говорящим) нативно, поэтому
эту задачу решает отдельный проход через LLM: Claude (через OpenRouter)
читает уже готовую расшифровку с тайм-кодами и размечает, кто именно
говорит в каждой реплике — по смыслу разговора (кто представляется
официантом, кто делает заказ, и т.д.), а не по голосовым характеристикам.

Для длинных расшифровок (60-90 минут) текст режется на части. Части
обрабатываются ПАРАЛЛЕЛЬНО для скорости, кроме первой — первый чанк
обрабатывается отдельно и задаёт "словарь ролей" (кто "Официант 1", кто
"Гость 2" и т.д.), который передаётся как контекст во все остальные чанки.
Без этого шага параллельная обработка рискует разъехаться по нумерации
ролей между частями (один и тот же официант может стать "Официант 1" в
одном куске и "Официант 2" в другом, если обрабатывать части независимо
без общего источника контекста).
"""

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-5"

MAX_PARALLEL_CHUNKS = 5

DIARIZATION_SYSTEM_PROMPT = """Ты помогаешь разметить расшифровку аудиозаписи визита тайного гостя в ресторан.

На вход ты получаешь текст с тайм-кодами, но БЕЗ разметки говорящих — просто
последовательность реплик. Твоя задача — определить, кто говорит каждую
реплику, ПО СМЫСЛУ разговора (кто представляется сотрудником, кто делает
заказ, кто говорит "Здравствуйте, меня зовут..." — это сотрудник, и т.д.).

Роли, которые нужно различать: "Гость", "Официант", "Хостес", "Бармен",
"Менеджер", "Другой сотрудник". Если несколько гостей — "Гость 1", "Гость 2".
Если несколько официантов — "Официант 1", "Официант 2". Если по контексту
непонятно, кто говорит (обрывок фразы, шум, посторонний разговор не по теме
визита) — помечай как "Неразборчиво" или "Посторонний разговор", не гадай.

ВАЖНО: часть текста может быть посторонним разговором, не относящимся к
визиту в ресторан (гости обсуждают свои дела, планы, другие темы). Такие
фрагменты тоже нужно включить в вывод, просто пометь их как "Гость"
(с сохранением тайм-кода) — не нужно их вырезать, но и не нужно пытаться
притянуть их к теме ресторана.

Формат вывода — строго построчно, без преамбулы и пояснений:
[тайм-код] Роль: текст реплики

Пример:
[00:17] Официант: Здравствуйте, меня зовут Дмитрий, я буду вас обслуживать.
[00:22] Гость: Добрый день, а что у вас есть из стейков?

Сохраняй тайм-коды из исходного текста без изменений. Не переписывай и не
исправляй сам текст реплик — только добавляй роль перед каждой строкой.
Если несколько строк подряд явно принадлежат одному говорящему без ответа
собеседника — можно объединить их под одной ролью, но не меняй содержание."""

CONTINUATION_CONTEXT_TEMPLATE = """

ВАЖНО ДЛЯ СОГЛАСОВАННОСТИ РОЛЕЙ: это продолжение расшифровки, а не её начало.
В начале визита уже были определены следующие роли: {roles_list}
Используй ТЕ ЖЕ обозначения для тех же людей, если по контексту понятно, что
это те же говорящие (например, тот же официант, который представился в
начале). Не создавай новых номеров ("Официант 2") для уже известных людей —
только если по контексту это явно другой, новый человек."""


class DiarizationError(Exception):
    pass


def _call_openrouter(system_prompt: str, user_message: str, api_key: str, max_tokens: int = 16000) -> str:
    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices", [])
        if not choices:
            raise DiarizationError(f"Пустой ответ от OpenRouter (нет choices): {result}")
        content = choices[0].get("message", {}).get("content")
        if not content:
            finish_reason = choices[0].get("finish_reason")
            raise DiarizationError(
                f"OpenRouter вернул пустой content при диаризации (finish_reason={finish_reason})"
            )
        return content
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise DiarizationError(f"Ошибка OpenRouter при диаризации ({e.code}): {body[:300]}")


def _split_into_chunks(plain_transcript: str, max_chars_per_chunk: int = 5000) -> list[str]:
    lines = plain_transcript.split("\n")
    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        if current_length + len(line) > max_chars_per_chunk and current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_length = 0
        current_chunk_lines.append(line)
        current_length += len(line)

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks


def _extract_roles(diarized_text: str) -> list[str]:
    """Извлекает уникальный список ролей (в порядке первого появления) из
    уже размеченного текста вида '[00:17] Официант: текст'."""
    roles = []
    seen = set()
    for line in diarized_text.split("\n"):
        match = re.match(r"\[[\d:]+\]\s*([^:]+):", line)
        if match:
            role = match.group(1).strip()
            if role not in seen and role not in ("Неразборчиво", "Посторонний разговор"):
                seen.add(role)
                roles.append(role)
    return roles


def diarize_transcript(plain_transcript: str, api_key: str) -> str:
    """Размечает по говорящим текст без диаризации (от Groq Whisper).

    Для длинных расшифровок: первый чанк обрабатывается отдельно (задаёт
    словарь ролей), остальные чанки — параллельно, с передачей списка ролей
    из первого чанка для согласованности нумерации говорящих."""
    chunks = _split_into_chunks(plain_transcript)

    if len(chunks) == 1:
        user_message = f"Вот расшифровка для разметки по говорящим:\n\n{chunks[0]}"
        return _call_openrouter(DIARIZATION_SYSTEM_PROMPT, user_message, api_key).strip()

    # Первый чанк — отдельно, он задаёт словарь ролей для всех остальных.
    first_user_message = f"Вот фрагмент расшифровки для разметки по говорящим:\n\n{chunks[0]}"
    first_result = _call_openrouter(DIARIZATION_SYSTEM_PROMPT, first_user_message, api_key).strip()

    roles = _extract_roles(first_result)
    roles_list = ", ".join(roles) if roles else "(роли не удалось определить чётко)"
    continuation_system_prompt = DIARIZATION_SYSTEM_PROMPT + CONTINUATION_CONTEXT_TEMPLATE.format(
        roles_list=roles_list
    )

    def _process_remaining_chunk(chunk: str) -> str:
        user_message = f"Вот фрагмент расшифровки для разметки по говорящим:\n\n{chunk}"
        return _call_openrouter(continuation_system_prompt, user_message, api_key).strip()

    remaining_chunks = chunks[1:]
    remaining_results = [None] * len(remaining_chunks)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
        futures_to_index = {
            executor.submit(_process_remaining_chunk, chunk): index
            for index, chunk in enumerate(remaining_chunks)
        }
        for future in futures_to_index:
            index = futures_to_index[future]
            remaining_results[index] = future.result()

    all_parts = [first_result] + remaining_results
    return "\n".join(all_parts)
