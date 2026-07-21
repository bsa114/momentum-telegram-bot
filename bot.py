"""
Бот для Render.com (Web Service, бесплатный тариф + keep-alive).
Параллельно с polling-ботом работает лёгкий HTTP-сервер для health-check,
чтобы Render не "усыплял" сервис — совместно с внешним пингом (UptimeRobot).
Текст обрабатывает Dify (OpenRouter/Groq внутри), голос — DashScope.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import urllib.error
import urllib.request

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import CommandStart
from aiogram.types import Message
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


@dp.message(CommandStart())
async def cmd_start(message: Message):
    conversation_ids.pop(message.chat.id, None)
    await message.answer(
        "Привет! Я на связи, работаю быстро и понимаю текст и голосовые. "
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


@dp.message(F.text)
async def handle_text(message: Message):
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, ask_dify, message.text, message.chat.id)
    await message.answer(answer)


@dp.message()
async def handle_other(message: Message):
    await message.answer("Пока умею обрабатывать только текст и голосовые.")


async def health_check(request):
    """Endpoint для проверки живости сервиса — сюда будет стучаться UptimeRobot."""
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
