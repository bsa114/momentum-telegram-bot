"""
Бот для Render.com (Web Service, бесплатный тариф + keep-alive).
Параллельно с polling-ботом работает лёгкий HTTP-сервер для health-check,
чтобы Render не "усыплял" сервис — совместно с внешним пингом (cron-job.org).
Текст обрабатывает Dify (OpenRouter/Groq внутри), голос и фото (OCR) — DashScope.
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

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DIFY_API_KEY = os.environ.get("DIFY_API_KEY")
DIFY_BASE_URL = os.environ.get("DIFY_BASE_URL", "https://api.dify.ai/v1")
PORT = int(os.environ.get("PORT", 10000))

if not BOT_TOKEN or not DASHSCOPE_API_KEY or not DIFY_API_KEY:
    raise RuntimeError(
        "Не заданы TELEGRAM_BOT_TOKEN, DASHSCOPE_API_KEY и/или DIFY_API_KEY"
    )

DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
ASR_MODEL = "qwen3-asr-flash"
OCR_MODEL = "qwen-vl-ocr"
DIFY_CHAT_URL = f"{DIFY_BASE_URL}/chat-messages"

logging.basicConfig(level=logging.INFO)

session = AiohttpSession(timeout=60)
bot = Bot(token=BOT_TOKEN, session=session)
dp = Dispatcher()

# Храним conversation_id для каждого чата — так Dify помнит контекст диалога.
# Простое хранилище в памяти процесса: сбрасывается при перезапуске сервиса,
# этого достаточно для обычной работы бота.
conversation_ids: dict[int, str] = {}


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
        # zoom ~2x повышает разрешение картинки — так текст распознаётся точнее
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image_bytes = pixmap.tobytes("png")

        page_text = extract_text_from_image(image_bytes, "page.png")
        if page_text.startswith("[Ошибка"):
            recognized_pages.append(f"--- Страница {page_index + 1} ---\n{page_text}")
        else:
            recognized_pages.append(f"--- Страница {page_index + 1} ---\n{page_text}")

    pdf.close()

    result = "\n\n".join(recognized_pages)
    if total_pages > max_pages:
        result += f"\n\n[Обработаны первые {max_pages} страниц из {total_pages}]"

    return result


@dp.message(CommandStart())
async def cmd_start(message: Message):
    conversation_ids.pop(message.chat.id, None)
    await message.answer(
        "Привет! Я на связи, работаю быстро и понимаю текст и голосовые. "
        "Также умею распознавать текст с фото и отсканированных PDF — просто пришли файл. "
        "Помню контекст нашего разговора — если захочешь начать с чистого листа, напиши /reset."
    )


@dp.message(F.text == "/reset")
async def cmd_reset(message: Message):
    conversation_ids.pop(message.chat.id, None)
    await message.answer("Память диалога сброшена, начинаем с чистого листа.")


@dp.message(F.voice)
async def handle_voice(message: Message):
    await message.answer("Слушаю голосовое сообщение...")
    file = await bot.get_file(message.voice.file_id)
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
    answer = await loop.run_in_executor(None, ask_dify, transcript, message.chat.id)
    await message.answer(answer)


@dp.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("Распознаю текст на фото...")
    photo = message.photo[-1]  # берём версию с наибольшим разрешением
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
    file_name = message.document.file_name or ""
    mime_type = message.document.mime_type or ""

    if not (file_name.lower().endswith(".pdf") or mime_type == "application/pdf"):
        await message.answer("Пока умею распознавать текст только из PDF-файлов.")
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


@dp.message(F.text)
async def handle_text(message: Message):
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask_dify, message.text, message.chat.id)
    await message.answer(answer)


@dp.message()
async def handle_other(message: Message):
    await message.answer("Пока умею обрабатывать только текст, голосовые и фото.")


async def health_check(request):
    """Endpoint для проверки живости сервиса — сюда стучится cron-job.org."""
    return web.Response(text="Bot is alive")


async def start_web_server():
    """Лёгкий HTTP-сервер, чтобы Render считал сервис 'веб-сервисом'."""
    app = web.Application()
    app.router.add_get("/", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logging.info(f"HTTP-сервер для health-check запущен на порту {PORT}")


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
    await run_bot()


if __name__ == "__main__":
    asyncio.run(main())
