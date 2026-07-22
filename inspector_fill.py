"""
Заполняет xlsx-анкету на основе расшифровки визита, используя Claude (через
OpenRouter — тот же провайдер, что уже подключён в Dify) и системный промпт
"Инспектора" (тот же, что используется в Claude.ai проекте).

Модель не трогает файл напрямую — она возвращает строгий JSON вида
{"filled": [{"row": N, "score": X, "comment": "..."}], "questions": [...]},
а запись в реальные ячейки xlsx делает код (openpyxl), не модель.
Это надёжнее, чем полагаться на то, что модель сама аккуратно работает с файлом.
"""

import json
import os
import urllib.error
import urllib.request

from xlsx_structure import build_compact_template_text, extract_template_structure

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-sonnet-5"

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
with open(os.path.join(PROMPT_DIR, "inspector_agent_v2.md"), "r", encoding="utf-8") as f:
    INSPECTOR_SYSTEM_PROMPT = f.read()

OUTPUT_FORMAT_INSTRUCTIONS = """

## ВАЖНОЕ ДОПОЛНЕНИЕ К ФОРМАТУ ОТВЕТА (для автоматической обработки)

Ты работаешь не в интерактивном чате, а как часть автоматического Telegram-бота.
Поэтому вместо обычного диалога с уточняющими вопросами по одному, ты должен
вернуть ОДИН СТРОГИЙ JSON-объект и ничего больше — ни преамбулы, ни пояснений,
ни markdown-разметки вокруг JSON.

Формат ответа:
{
  "header": {"restaurant": "...", "date": "...", "waiter_name": "...", "waiter_description": "...", "visit_type": "...", "time_in": "...", "time_out": "..."},
  "filled": [
    {"row": 34, "score": 2, "comment": "текст комментария инспектора"}
  ],
  "questions": [
    {
      "section": "название раздела",
      "topic": "что нужно уточнить",
      "context": "что видно из расшифровки по этому поводу",
      "options": [
        {"label": "1", "row": 40, "score": 3, "comment": "готовый комментарий для этого варианта"},
        {"label": "2", "row": 41, "score": 0, "comment": "готовый комментарий для этого варианта"}
      ]
    }
  ]
}

Правила:
- "filled" — все пункты анкеты, которые ты можешь закрыть уверенно по правилам фазы Б.
  Указывай ТОЧНУЮ строку (row) из структуры анкеты, которая соответствует выбранной
  позиции шкалы — не первую строку блока, а именно ту, что выбрана.
- "questions" — все пункты, которые в обычной работе ты вынес бы в фазу В (уточнения).
  Для каждого варианта ответа заранее укажи row и score, которые нужно проставить,
  если пользователь выберет этот вариант.
- Не включай в ответ никакого текста вне JSON.
- Строго следуй техническим правилам записи (балл в нужную строку блока, не в первую),
  правилам жанра комментариев (прошедшее время, без имён, без оценочных слов,
  фактологично, с цитатами) — они остаются в силе, просто результат подаётся как JSON,
  а не как текст в чат.
"""


class InspectorFillError(Exception):
    pass


def _call_claude_via_openrouter(
    system_prompt: str, user_message: str, api_key: str, max_tokens: int = 8000
) -> str:
    """Вызывает Claude через OpenRouter — формат OpenAI chat completions,
    не нативный Anthropic Messages API (у OpenRouter system идёт как
    отдельное сообщение в messages, а не отдельным полем)."""
    payload = json.dumps(
        {
            "model": OPENROUTER_MODEL,
            "max_tokens": max_tokens,
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
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices", [])
        if not choices:
            raise InspectorFillError(f"Пустой ответ от OpenRouter: {result}")
        return choices[0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise InspectorFillError(f"Ошибка OpenRouter ({e.code}): {body[:300]}")


def _extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели, даже если она обернула его в markdown-код."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise InspectorFillError(f"Не удалось разобрать JSON от модели: {e}\nОтвет модели: {text[:500]}")


def analyze_transcript_and_fill(
    empty_template_path: str,
    transcript_text: str,
    intro_context: str,
    openrouter_api_key: str,
) -> dict:
    """Первый проход: разбор расшифровки, заполнение уверенных пунктов,
    формирование списка уточняющих вопросов."""
    structure = extract_template_structure(empty_template_path)
    template_text = build_compact_template_text(structure)

    user_message = f"""Вот структура пустой анкеты (структурный источник правды):

{template_text}

Вводная информация о визите: {intro_context}

Расшифровка визита:

{transcript_text}

Заполни анкету по правилам из системного промпта, верни результат строго в формате JSON, описанном в дополнении к формату ответа."""

    full_system_prompt = INSPECTOR_SYSTEM_PROMPT + OUTPUT_FORMAT_INSTRUCTIONS

    raw_response = _call_claude_via_openrouter(full_system_prompt, user_message, openrouter_api_key)
    parsed = _extract_json(raw_response)
    parsed["_structure"] = structure
    return parsed


def resolve_questions_answers(
    fill_result: dict,
    user_answers: dict[int, str],
) -> list[dict]:
    """Превращает ответы пользователя (номер вопроса -> выбранный label) в
    финальный список {"row": ..., "score": ..., "comment": ...} для дозаписи в xlsx."""
    resolved = []
    questions = fill_result.get("questions", [])

    for index, question in enumerate(questions, start=1):
        answer_label = user_answers.get(index)
        if answer_label is None:
            continue
        for option in question.get("options", []):
            if option.get("label") == answer_label:
                resolved.append(
                    {"row": option["row"], "score": option["score"], "comment": option["comment"]}
                )
                break

    return resolved
