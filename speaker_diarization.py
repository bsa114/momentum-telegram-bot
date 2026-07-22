"""
Groq Whisper не даёт диаризацию (разделение по говорящим) нативно, поэтому
эту задачу решает отдельный проход через LLM: Claude (через OpenRouter)
читает уже готовую расшифровку с тайм-кодами и размечает, кто именно
говорит в каждой реплике — по смыслу разговора (кто представляется
официантом, кто делает заказ, и т.д.), а не по голосовым характеристикам.

Это отдельный, более простой и дешёвый запрос, чем основной запрос
заполнения анкеты — здесь не нужен весь системный промпт Инспектора.
"""

import json
import urllib.error
import urllib.request

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-5"

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


def diarize_transcript(plain_transcript: str, api_key: str) -> str:
    """Размечает по говорящим текст без диаризации (от Groq Whisper).
    Для очень длинных расшифровок (60-90 минут) разбивает на части,
    чтобы не упереться в лимит контекста и вывода одного запроса."""
    max_chars_per_chunk = 15000  # с запасом относительно лимита вывода модели
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

    diarized_parts = []
    for chunk in chunks:
        user_message = f"Вот фрагмент расшифровки для разметки по говорящим:\n\n{chunk}"
        result = _call_openrouter(DIARIZATION_SYSTEM_PROMPT, user_message, api_key)
        diarized_parts.append(result.strip())

    return "\n".join(diarized_parts)
