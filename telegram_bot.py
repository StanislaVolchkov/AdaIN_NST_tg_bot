# -*- coding: utf-8 -*-
"""

Telegram Bot

"""
#!pip install aiogram
#!pip install nest-asyncio
import asyncio
from aiohttp import web
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook
#import nest_asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from models import nst_model
import zipfile

TOKEN = "5163951677:AAGYMXeWn-3RsQ31XOL8MPrHegxc_77EoRQ"
CONNECTION_TYPE = 'WEBHOOK'
WEBHOOK_HOST = 'https://my-neural-style-transfer.herokuapp.com'
WEBHOOK_PATH = f'/bot'
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Configure logging
logging.basicConfig(level=logging.INFO)

#создаем самого бота
loop = asyncio.get_event_loop() 
storage = MemoryStorage()
bot = Bot(token=TOKEN, loop=loop)
dp = Dispatcher(bot, storage=storage)
logging.basicConfig(level=logging.INFO)

wth_zip = 'models_wth/weights.pth.zip'
with zipfile.ZipFile(wth_zip, 'r') as zip_file:
    zip_file.extract('weights.pth')

# избавление от вложенных циклов запуска ядра
#nest_asyncio.apply()

#async def main():
#    print("begin")
#    response = await asyncio.sleep(2)
#    print("the end", response)

#loop.run_until_complete(main())

# добавление клавиатуры
kb = ReplyKeyboardMarkup(resize_keyboard=True)

b1 = KeyboardButton('/Стилизация')
b2 = KeyboardButton('Помощь')
b3 = KeyboardButton('Разработчику')

kb.add(b1).row(b2, b3)

# определение декораторов добавляющих функционал боту
async def send_welcome(msg: types.Message):
    await msg.answer(f'Я бот по переносу стиля. Приятно познакомиться, {msg.from_user.first_name}', reply_markup=kb)

async def get_help_info(msg: types.Message):
   if msg.text.lower() == 'помощь':
       await msg.answer('Хорошо, я покажу, как это делается')
   elif msg.text.lower() == 'разработчику':
       await msg.answer('Сейчас узнаем архитектуру!')
   else:
       await msg.answer('Не понимаю, что это значит.')

# Результат работы модели
async def getting_model_output(msg, state):
  await msg.answer(f"Все будет в лучшем виде через пару секунд!")
  async with state.proxy() as data:
    result = nst_model.get_transfer(data['content'], data['style'], data['percent'], 'weights.pth')
  await bot.send_photo(msg.chat.id, photo=result)

# задаем админку для осмысленного получения сообщений
class FSMAdmin(StatesGroup):
  content = State()
  style = State()
  percent = State()

async def start_transfer(msg: types.Message):
  await FSMAdmin.content.set()
  await msg.answer('Давайте выберем изображение, которое будем стилизовать!')

async def get_content_to_transfer(msg: types.Message, state=FSMContext):
  async with state.proxy() as data:
    image = msg.photo[-1]
    file_info = await bot.get_file(image.file_id)
    data['content'] = await bot.download_file(file_info.file_path)

  await FSMAdmin.next()
  await msg.answer('Теперь нужно загрузить изображение, стиль которого нужно перенести!')

async def get_style_to_transfer(msg: types.Message, state=FSMContext):
  async with state.proxy() as data:
    image = msg.photo[-1]
    file_info = await bot.get_file(image.file_id)
    data['style'] = await bot.download_file(file_info.file_path)

  await FSMAdmin.next()
  await msg.answer('Введите сколько % стиля вы хотите перенести (от 0 до 100)')

async def get_percent_to_transfer(msg: types.Message, state=FSMContext):
  async with state.proxy() as data:
    data['percent'] = int(msg.text)
  await getting_model_output(msg, state)
  
  await state.finish()

# заполняет обработчиками наш диспетчер
# сначала команды, потом текста
def all_handlers(dp: Dispatcher):
  dp.register_message_handler(send_welcome, commands=['start'])
  dp.register_message_handler(start_transfer, commands=['Стилизация'], state=None)
  dp.register_message_handler(get_content_to_transfer, content_types=['photo'], state=FSMAdmin.content)
  dp.register_message_handler(get_style_to_transfer, content_types=['photo'], state=FSMAdmin.style)
  dp.register_message_handler(get_percent_to_transfer, content_types=['text'], state=FSMAdmin.percent)
  dp.register_message_handler(get_help_info, content_types=['text'])
  
all_handlers(dp)

# запуск самого бота
async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    logging.warning("Hello!")
    
async def on_shutdown(dp):
    logging.warning("Shutting down...")
    await dp.storage.close()
    await dp.storage.wait_closed()
    logging.warning("Bye!")

if __name__ == '__main__':

    webhook_settings = False if CONNECTION_TYPE == 'POLLING' else True
    if webhook_settings:
        WEBAPP_PORT = 443
        WEBAPP_HOST = 'localhost'
        start_webhook(
            dispatcher=dp,
            webhook_path=WEBHOOK_PATH,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            skip_updates=True,
            host=WEBAPP_HOST,
            port=WEBAPP_PORT,
        )
    else:
        executor.start_polling(dp, skip_updates=True)

