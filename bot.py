"""
Momentum — единый Telegram-бот на Render.com (Web Service, бесплатный тариф + keep-alive).

Два режима работы в одном боте:

1. ОБЫЧНЫЙ РЕЖИМ (по умолчанию):
   - Текст -> Dify (chatflow, OpenRouter/Groq внутри)
   - Голос -> DashScope (qwen3-asr-flash) -> расшифровка -> Dify
   - Фото -> DashScope OCR (qwen-vl-ocr) -> файл с распознанным текстом
   - PDF -> постранично через тот же OCR -> файл с распознанным текстом

2. РЕЖИМ ИНСПЕКТОРА (включается автоматически при получении xlsx-файла):
   - xlsx-анкета -> сохраняется как шаблон визита
   - Аудио визита (60-90 мин, файлом до 20 МБ или ссылкой на Яндекс.Диск/
     Google Drive для больших записей) -> DashScope (fun-asr, диаризация,
     тайм-коды) -> расшифровка -> Claude API (промпт "Инспектора") ->
     заполненная xlsx-анкета, с уточняющими вопросами по неоднозначным пунктам

Оба режима используют общий HTTP-сервер для health-check (keep-alive через
cron-job.org) и для временной раздачи больших аудио из памяти в DashScope.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import urllib.error
import urllib.request

import fitz  # PyMuPDF

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message
from aiohttp import web

from inspector_fill import InspectorFillError, analyze_transcript_and_fill, resolve_questions_answers
from inspector_transcription import TranscriptionError, transcribe_long_audio
from link_download import LinkDownloadError, download_file_from_link
from temp_file_server import periodic_cleanup_task, pop_file_from_memory, serve_temp_file, store_file_in_memory
from xlsx_writer import apply_fill_results, check_for_formula_errors

# --- Переменные окружения ---

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY")
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL")  # напр. https://momentum-telegram-bot.onrender.com
PORT = int(os.environ.get("PORT", 10000))

if not BOT_TOKEN or not DASHSCOPE_API_KEY or not DIFY_API_KEY:
    raise RuntimeError(
        "Не заданы TELEGRAM_BOT_TOKEN, DASHSCOPE_API_KEY и/или DIFY_API_KEY"
    )

if not OPENROUTER_API_KEY or not PUBLIC_BASE_URL:
    raise RuntimeError(
        "Не заданы OPENROUTER_API_KEY и/или PUBLIC_BASE_URL — они нужны для режима Инспектора "
        "(заполнение xlsx-анкет по аудио визита)"
    )

DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
ASR_MODEL = "qwen3-asr-flash"
OCR_MODEL = "qwen-vl-ocr"
DIFY_CHAT_URL = f"{DIFY_BASE_URL}/chat-messages"

logging.basicConfig(level=logging.INFO)

session = AiohttpSession(timeout=60)
bot = Bot(token=BOT_TOKEN, session=session)
dp = Dispatcher()

# Обычный режим: conversation_id для Dify по каждому чату.
conversation_ids: dict[int, str] = {}

# Режим Инспектора: состояние диалога по каждому чату.
# Поля: stage ("awaiting_audio" | "awaiting_answers"), template_path,
# transcript_text, fill_result.
inspector_states: dict[int, dict] = {}


def _in_inspector_mode(chat_id: int) -> bool:
    return chat_id in inspector_states


def _reset_inspector_state(chat_id: int) -> None:
    inspector_states.pop(chat_id, None)


# ============================================================
# ОБЫЧНЫЙ РЕЖИМ — Dify (текст) + DashScope (голос/фото/PDF)
# ============================================================


def ask_dify(user_text: str, chat_id: int) -> str:
    """Отправляет сообщение в Dify. Передаёт сохранённый conversation_id,
    чтобы Dify помнил контекст диалога с этим пользователем."""
    payload_dict = {
        "inputs": {},
        "query": user_text,
        "response_mode": "blocking",
        "user": f"telegram-{chat_id}",
    }

    existing_conversation_id = conversation_ids.get(chat_id)
    if existing_conversation_id:
        payload_dict["conversation_id"] = existing_conversation_id

    payload = json.dumps(payload_dict).encode("utf-8")

    req = urllib.request.Request(
        DIFY_CHAT_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DIFY_API_KEY}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            result = json.loads(response.read().decode("utf-8"))

        new_conversation_id = result.get("conversation_id")
        if new_conversation_id:
            conversation_ids[chat_id] = new_conversation_id

        return result.get("answer", "Пустой ответ от Dify.")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"Ошибка Dify ({e.code}): {body[:200]}"
    except Exception as e:
        return f"Произошла ошибка при обращении к Dify: {e}"


def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    mime_type = mimetypes.guess_type(filename)[0] or "audio/ogg"
    base64_str = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_str}"

    payload = json.dumps(
        {
            "model": ASR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": data_uri}}],
                }
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[Ошибка распознавания речи: {e.code} {body[:150]}]"
    except Exception as e:
        return f"[Ошибка распознавания речи: {e}]"


def extract_text_from_image(file_bytes: bytes, filename: str) -> str:
    """Распознаёт весь текст на изображении через DashScope (qwen-vl-ocr)."""
    mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    base64_str = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_str}"

    payload = json.dumps(
        {
            "model": OCR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {
                            "type": "text",
                            "text": "Распознай весь текст на изображении и выведи только его, "
                                    "без каких-либо комментариев и пояснений от себя.",
                        },
                    ],
                }
            ],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[Ошибка распознавания текста: {e.code} {body[:150]}]"
    except Exception as e:
        return f"[Ошибка распознавания текста: {e}]"


def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 30) -> str:
    """Рендерит каждую страницу PDF (в т.ч. отсканированного, без текстового слоя)
    в изображение и прогоняет через ту же OCR-модель, что и обычные фото."""
    try:
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return f"[Ошибка чтения PDF: {e}]"

    total_pages = pdf.page_count
    if total_pages == 0:
        return "[В этом PDF нет страниц]"

    pages_to_process = min(total_pages, max_pages)
    recognized_pages = []

    for page_index in range(pages_to_process):
        page = pdf.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pixmap.tobytes("png")

        page_text = extract_text_from_image(image_bytes, "page.png")
        recognized_pages.append(f"--- Страница {page_index + 1} ---\n{page_text}")

    pdf.close()

    result = "\n\n".join(recognized_pages)
    if total_pages > max_pages:
        result += f"\n\n[Обработаны первые {max_pages} страниц из {total_pages}]"

    return result


# ============================================================
# РЕЖИМ ИНСПЕКТОРА — аудио визита -> заполненная xlsx-анкета
# ============================================================


def _parse_answers(text: str) -> dict[int, str]:
    """Разбирает ответ вида '1-3, 2-1, 3-4' в {1: '3', 2: '1', 3: '4'}."""
    result = {}
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    for part in parts:
        if "-" not in part:
            continue
        question_num, label = part.split("-", 1)
        question_num = question_num.strip()
        label = label.strip()
        if question_num.isdigit():
            result[int(question_num)] = label
    return result


async def _inspector_process_audio_bytes(
    message: Message, chat_id: int, state: dict, file_bytes: bytes, content_type: str
) -> None:
    """Общая логика после того, как байты аудио визита уже получены —
    не важно, напрямую из Telegram или скачаны по ссылке."""
    file_id = store_file_in_memory(file_bytes, content_type)
    public_url = f"{PUBLIC_BASE_URL}/tmp-audio/{file_id}"

    loop = asyncio.get_event_loop()
    try:
        transcript_text = await loop.run_in_executor(
            None, transcribe_long_audio, public_url, DASHSCOPE_API_KEY
        )
    except TranscriptionError as e:
        await message.answer(f"Не получилось распознать аудио: {e}")
        return
    finally:
        pop_file_from_memory(file_id)

    if not transcript_text.strip():
        await message.answer("Не удалось получить текст из аудио — расшифровка пустая.")
        return

    state["transcript_text"] = transcript_text

    txt_bytes = transcript_text.encode("utf-8")
    document = BufferedInputFile(txt_bytes, filename="расшифровка.txt")
    await message.answer_document(document, caption="Расшифровка готова.")

    await message.answer("Разбираю расшифровку и заполняю анкету, подожди немного...")

    try:
        fill_result = await loop.run_in_executor(
            None,
            analyze_transcript_and_fill,
            state["template_path"],
            transcript_text,
            "Визит тайного гостя, аудио прислано через Telegram-бота.",
            OPENROUTER_API_KEY,
        )
    except InspectorFillError as e:
        await message.answer(f"Не получилось обработать анкету: {e}")
        return

    state["fill_result"] = fill_result

    questions = fill_result.get("questions", [])
    filled_count = len(fill_result.get("filled", []))

    if not questions:
        await _inspector_finalize_and_send(message, chat_id)
        return

    state["stage"] = "awaiting_answers"

    intro_line = (
        f"Заполнил {filled_count} пункт(ов) сам. Осталось уточнить {len(questions)} — отвечай одним "
        f"сообщением, номерами вариантов через запятую (например: 1-3, 2-1, 3-4)."
    )

    question_blocks = []
    for index, q in enumerate(questions, start=1):
        block_lines = [f"\n{index}. [{q.get('section', '')}] {q.get('topic', '')}"]
        if q.get("context"):
            block_lines.append(f"   Из расшифровки: {q['context']}")
        for option in q.get("options", []):
            block_lines.append(f"   {option['label']}) {option.get('comment', '')}")
        question_blocks.append("\n".join(block_lines))

    await _send_in_chunks(message, intro_line, question_blocks)


TELEGRAM_MESSAGE_SAFE_LIMIT = 3500


async def _send_in_chunks(message: Message, intro_line: str, blocks: list[str]) -> None:
    """Отправляет список текстовых блоков одним или несколькими сообщениями,
    не превышая безопасный лимит длины сообщения в Telegram (реальный лимит
    4096 символов, берём с запасом)."""
    current_chunk = intro_line
    is_first_chunk = True

    for block in blocks:
        candidate = current_chunk + "\n" + block
        if len(candidate) > TELEGRAM_MESSAGE_SAFE_LIMIT:
            await message.answer(current_chunk)
            current_chunk = block
            is_first_chunk = False
        else:
            current_chunk = candidate

    if current_chunk:
        await message.answer(current_chunk)


async def _inspector_finalize_and_send(message: Message, chat_id: int) -> None:
    state = inspector_states.get(chat_id)
    if not state:
        return

    fill_result = state["fill_result"]
    template_path = state["template_path"]
    output_path = f"/tmp/inspector_output_{chat_id}.xlsx"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        apply_fill_results,
        template_path,
        output_path,
        fill_result.get("header", {}),
        fill_result.get("filled", []),
    )

    problems = await loop.run_in_executor(None, check_for_formula_errors, output_path)

    with open(output_path, "rb") as f:
        output_bytes = f.read()

    document = BufferedInputFile(output_bytes, filename="заполненная_анкета.xlsx")

    caption_lines = [f"Готово! Заполнено пунктов: {len(fill_result.get('filled', []))}."]
    if problems:
        caption_lines.append(f"⚠️ Обнаружены проблемы с формулами: {', '.join(problems)}")

    await message.answer_document(document, caption="\n".join(caption_lines))
    await message.answer(
        "Режим Инспектора завершён. Можешь прислать новую xlsx-анкету для следующего визита, "
        "или просто писать/присылать файлы как обычно."
    )

    _reset_inspector_state(chat_id)


# ============================================================
# ОБРАБОТЧИКИ TELEGRAM
# ============================================================


@dp.message(CommandStart())
async def cmd_start(message: Message):
    conversation_ids.pop(message.chat.id, None)
    _reset_inspector_state(message.chat.id)
    await message.answer(
        "Привет! Я на связи, работаю быстро и понимаю текст и голосовые. "
        "Также умею распознавать текст с фото и отсканированных PDF — просто пришли файл.\n\n"
        "Отдельная возможность — заполнение чек-листа тайного гостя по аудиозаписи визита: "
        "пришли мне пустую xlsx-анкету, и я перейду в режим Инспектора (об этом расскажу "
        "подробнее в этот момент).\n\n"
        "Помню контекст нашего разговора — если захочешь начать с чистого листа, напиши /reset."
    )


@dp.message(F.text == "/reset")
async def cmd_reset(message: Message):
    conversation_ids.pop(message.chat.id, None)
    was_in_inspector = _in_inspector_mode(message.chat.id)
    _reset_inspector_state(message.chat.id)
    if was_in_inspector:
        await message.answer("Режим Инспектора прерван, память диалога сброшена. Начинаем с чистого листа.")
    else:
        await message.answer("Память диалога сброшена, начинаем с чистого листа.")


@dp.message(F.voice | F.audio)
async def handle_audio_message(message: Message):
    chat_id = message.chat.id

    # --- Режим Инспектора: аудио визита ---
    if _in_inspector_mode(chat_id):
        state = inspector_states[chat_id]
        if state.get("stage") != "awaiting_audio":
            await message.answer(
                "Сейчас жду ответы на уточняющие вопросы, а не новое аудио. "
                "Если хочешь начать визит заново, напиши /reset."
            )
            return

        file_obj = message.voice or message.audio
        await message.answer("Загружаю аудио и отправляю на распознавание, это может занять несколько минут...")

        try:
            file = await bot.get_file(file_obj.file_id)
        except Exception:
            await message.answer(
                "Telegram не даёт скачать этот файл напрямую — скорее всего, он больше 20 МБ "
                "(это ограничение самого Telegram, не наше). Загрузи аудио на Яндекс.Диск или "
                "Google Drive, сделай публичную ссылку и пришли её мне текстом."
            )
            return

        file_bytes_io = await bot.download_file(file.file_path)
        file_bytes = file_bytes_io.read()
        content_type = "audio/ogg" if message.voice else (file_obj.mime_type or "audio/mpeg")

        await _inspector_process_audio_bytes(message, chat_id, state, file_bytes, content_type)
        return

    # --- Обычный режим: голосовое сообщение через Dify ---
    await message.answer("Слушаю голосовое сообщение...")
    file_obj = message.voice or message.audio
    file = await bot.get_file(file_obj.file_id)
    file_bytes_io = await bot.download_file(file.file_path)
    file_bytes = file_bytes_io.read()

    loop = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(
        None, transcribe_audio, file_bytes, file.file_path
    )

    if transcript.startswith("[Ошибка"):
        await message.answer(transcript)
        return

    await message.answer(f"Распознал: «{transcript}»")
    answer = await loop.run_in_executor(None, ask_dify, transcript, chat_id)
    await message.answer(answer)


@dp.message(F.photo)
async def handle_photo(message: Message):
    # Фото не участвует в режиме Инспектора — всегда обычный OCR.
    await message.answer("Распознаю текст на фото...")
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes_io = await bot.download_file(file.file_path)
    file_bytes = file_bytes_io.read()

    loop = asyncio.get_event_loop()
    recognized_text = await loop.run_in_executor(
        None, extract_text_from_image, file_bytes, file.file_path
    )

    if recognized_text.startswith("[Ошибка"):
        await message.answer(recognized_text)
        return

    if not recognized_text:
        await message.answer("Не удалось найти текст на этом изображении.")
        return

    txt_bytes = recognized_text.encode("utf-8")
    document = BufferedInputFile(txt_bytes, filename="распознанный_текст.txt")
    await message.answer_document(document, caption="Готово! Вот распознанный текст.")


@dp.message(F.document)
async def handle_document(message: Message):
    chat_id = message.chat.id
    file_name = message.document.file_name or ""
    mime_type = message.document.mime_type or ""
    is_xlsx = file_name.lower().endswith(".xlsx")
    is_pdf = file_name.lower().endswith(".pdf") or mime_type == "application/pdf"

    # --- xlsx всегда запускает/продолжает режим Инспектора ---
    if is_xlsx:
        file = await bot.get_file(message.document.file_id)
        file_bytes_io = await bot.download_file(file.file_path)

        template_path = f"/tmp/inspector_template_{chat_id}.xlsx"
        with open(template_path, "wb") as f:
            f.write(file_bytes_io.read())

        inspector_states[chat_id] = {"stage": "awaiting_audio", "template_path": template_path}

        await message.answer(
            "Анкета получена — перехожу в режим Инспектора. Пришли аудиозапись визита "
            "(60–90 минут):\n"
            "• Файл до 20 МБ — как обычное аудио/голосовое.\n"
            "• Файл больше 20 МБ — загрузи на Яндекс.Диск или Google Drive, сделай "
            "публичную ссылку и пришли её текстом (можно сразу, без попытки прислать файл).\n\n"
            "Напиши /reset, если хочешь выйти из этого режима."
        )
        return

    # --- В режиме Инспектора документ — это либо не то (PDF/др.), либо ошибка ---
    if _in_inspector_mode(chat_id):
        await message.answer(
            "Сейчас в режиме Инспектора жду xlsx-анкету или аудио визита, а не этот файл. "
            "Напиши /reset, если хочешь выйти из режима Инспектора и вернуться к обычной работе."
        )
        return

    # --- Обычный режим: PDF OCR ---
    if not is_pdf:
        await message.answer(
            "Пока умею распознавать текст только из PDF-файлов, либо запускать режим "
            "Инспектора по xlsx-анкете."
        )
        return

    await message.answer("Распознаю текст в PDF, это может занять немного времени...")
    file = await bot.get_file(message.document.file_id)
    file_bytes_io = await bot.download_file(file.file_path)
    file_bytes = file_bytes_io.read()

    loop = asyncio.get_event_loop()
    recognized_text = await loop.run_in_executor(None, extract_text_from_pdf, file_bytes)

    if recognized_text.startswith("[Ошибка"):
        await message.answer(recognized_text)
        return

    txt_bytes = recognized_text.encode("utf-8")
    document = BufferedInputFile(txt_bytes, filename="распознанный_текст.txt")
    await message.answer_document(document, caption="Готово! Вот распознанный текст из PDF.")


@dp.message(F.text.regexp(r"https?://\S+"))
async def handle_link(message: Message):
    chat_id = message.chat.id

    # Ссылка имеет смысл только в режиме Инспектора (ссылка на большое аудио визита).
    if not _in_inspector_mode(chat_id):
        # В обычном режиме ссылка — просто текст для Dify.
        await handle_text(message)
        return

    state = inspector_states[chat_id]
    if state.get("stage") != "awaiting_audio":
        await message.answer(
            "Сейчас жду ответы на уточняющие вопросы, а не новую ссылку. "
            "Если хочешь начать визит заново, напиши /reset."
        )
        return

    await message.answer(
        "Скачиваю аудио по ссылке, это может занять некоторое время для больших файлов..."
    )

    loop = asyncio.get_event_loop()
    try:
        file_bytes = await loop.run_in_executor(None, download_file_from_link, message.text.strip())
    except LinkDownloadError as e:
        await message.answer(
            f"Не получилось скачать файл по ссылке: {e}\n\n"
            "Поддерживаются: публичная ссылка Яндекс.Диска (yadi.sk/... или disk.yandex.ru/...), "
            "Google Drive (drive.google.com/file/d/.../view) или прямая ссылка на файл."
        )
        return

    await _inspector_process_audio_bytes(message, chat_id, state, file_bytes, "audio/mpeg")


@dp.message(F.text)
async def handle_text(message: Message):
    chat_id = message.chat.id

    # --- Режим Инспектора: ответы на уточняющие вопросы ---
    if _in_inspector_mode(chat_id):
        state = inspector_states[chat_id]
        if state.get("stage") != "awaiting_answers":
            await message.answer(
                "В режиме Инспектора сейчас жду xlsx-анкету или аудио визита, а не текст. "
                "Напиши /reset, если хочешь выйти из режима."
            )
            return

        user_answers = _parse_answers(message.text)
        if not user_answers:
            await message.answer(
                "Не разобрал ответ. Формат: номер_вопроса-вариант, через запятую, "
                "например: 1-3, 2-1, 3-4"
            )
            return

        resolved_rows = resolve_questions_answers(state["fill_result"], user_answers)
        state["fill_result"]["filled"].extend(resolved_rows)

        await message.answer("Принято. Собираю финальный файл...")
        await _inspector_finalize_and_send(message, chat_id)
        return

    # --- Обычный режим: текст через Dify ---
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask_dify, message.text, chat_id)
    await message.answer(answer)


@dp.message()
async def handle_other(message: Message):
    await message.answer("Пока умею обрабатывать только текст, голосовые, фото и документы (PDF/xlsx).")


# ============================================================
# HTTP-сервер (health-check + временная раздача аудио) и запуск бота
# ============================================================


async def health_check(request):
    """Endpoint для проверки живости сервиса — сюда стучится cron-job.org."""
    return web.Response(text="Bot is alive")


async def start_web_server():
    """Лёгкий HTTP-сервер: health-check + временная раздача аудио для DashScope."""
    app = web.Application()
    app.router.add_get("/", health_check)
    app.router.add_get("/tmp-audio/{file_id}", serve_temp_file)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logging.info(f"HTTP-сервер запущен на порту {PORT}")


async def run_bot():
    attempt = 0
    while True:
        try:
            attempt += 1
            logging.info(f"Бот запускается в режиме polling (попытка {attempt})...")
            if attempt == 1:
                await bot.delete_webhook(drop_pending_updates=True)
            await dp.start_polling(bot)
            break
        except Exception as e:
            wait = min(30, 5 * attempt)
            logging.warning(f"Сбой соединения: {e}. Повтор через {wait} сек...")
            await asyncio.sleep(wait)


async def main():
    await start_web_server()
    asyncio.create_task(periodic_cleanup_task())
    await run_bot()


if __name__ == "__main__":
    asyncio.run(main())
