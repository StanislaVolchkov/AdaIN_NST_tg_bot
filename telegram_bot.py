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
import os
from urllib.parse import urljoin

TOKEN = "5163951677:AAGYMXeWn-3RsQ31XOL8MPrHegxc_77EoRQ"
CONNECTION_TYPE = 'WEBHOOK'
WEBHOOK_HOST = 'https://my-neural-style-transfer.herokuapp.com'
WEBHOOK_PATH = f'/'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO)

#создаем самого бота
loop = asyncio.get_event_loop() 
storage = MemoryStorage()
bot = Bot(token=TOKEN, loop=loop, parse_mode=types.ParseMode.MARKDOWN)
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
kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

b1 = KeyboardButton('/Стилизация\U0001F311')
b2 = KeyboardButton('/2 в 1\U0001F315')
b3 = KeyboardButton('Примеры')
b4 = KeyboardButton('Помощь')
b5 = KeyboardButton('Разработчику')
b6 = KeyboardButton('Обратная связь\U0001F64B')
b7 = KeyboardButton('Github')

kb.add(b1, b2).row(b3, b4).row(b5, b6).add(b7)

kb2 = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

b1 = KeyboardButton('Готовые стили')

kb2.row(b1)

kb3 = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)

b1 = KeyboardButton('Эдвард Мунк "Крик"')
b2 = KeyboardButton('Фанри Матис "Женщина с шляпой"')
b3 = KeyboardButton('Виллем де Кунинг "Эшвилл"')
b4 = KeyboardButton('Пит Модриан "Композиция в коричневом и сером"')
b5 = KeyboardButton('Набросок женщины карандашом')

kb3.add(b1).add(b2).add(b3).add(b4).add(b5)

# определение декораторов добавляющих функционал боту
async def send_welcome(msg: types.Message):
    await msg.answer(f'Ас-саля́му але́йкум, {msg.from_user.first_name}\U0001F64F. Я бот по *переносу стиля*', reply_markup=kb)

async def get_help_info(msg: types.Message):

   if msg.text.lower() == 'помощь':
       await msg.answer('*Хорошо, я покажу, как это делается.*\n\n\
\U0001F313 *Как выбрать подходящий алгоритм?*\n \
— Если ты хочешь перенести ярко выраженный стиль (картины, текстуры, цвета) с одной фотографии на другую \
воспользуйся пунктом "Стилизация" и следуй дальнейшим инструкциям. \n \
— Если ты хочешь изменить содержимое фотографии, объединив с содержимым другой \
воспользуйся пунктом "2 в 1" и следуй дальнейшим инструкциям. \n\n\
\U000023F3 *Сколько времени ждать результат?*\n \
— "Стилизация" занимает до 10 секунд. \n \
— "2 в 1" находится в разработке. \n\n\
\U0001F5FF *Что делать, если я недоволен результатом?*\n \
— В данном случае стоит понимать, что алгоритмы не могут быть универсальными \
и некоторые изображения будут плохо переноситься на другие. Попробуйте \
немного изменить изображения стиля. Или воспользуйтесь экземпляром из "Готовые стили", \
каждый из которых дает хороший результат переноса.\n\
*Одна мудрость*\U0000261D: Помните, нельзя сделать белое из черного, иначе это будет просто белое.\n\n\
\U0001F306 *Какого размера фотографии принимает модель?*\n \
— Модель построенна на свертках, поэтому способна обрабатывать изображения любого разрешения. \
Но предпочтительным является размер больше, чем 512х512 во избежании потери качества при растягивании.\n\n\
\U0001F440 *Что лежит в основе моделей?*\n \
— Для получения информации об архитектуре перейдите в раздел "Разработчику".',  reply_markup=kb)
       
   elif msg.text.lower() == 'разработчику':
       await msg.answer('*Сейчас узнаем архитектуру!*\U0001F192\n\n\
*Стилизация*\U0001F311 \n \
  Алгоритм работает на основе последнего достижения в области Neural Style Transfer(без учета продвинутых методов). \
Модель AdaIN лишена ограничений в доступных к переносу стилях, при этом имея \
преимущество в скорости по сравнению с алгоритмом Gatys(5 сек против 15 мин). \
Главной особенностью данной модели является инновационный слой Adaptive Instance Normalization \
между слоями "енкодера" и "декодера", которые являются частями VGG-19 без bn.\n\n\
*2 в 1*\U0001F315 \n \
  Раздел в разработке.\n\n\
*Больше подробностей можно узнать на Github автора.*',  reply_markup=kb)
       
   elif msg.text.lower() == 'обратная связь\U0001F64B':
       await msg.answer('<b>С автором можно связаться:</b> @quality_pleasure\U0001F525 ', parse_mode='HTML' , reply_markup=kb)

   elif msg.text.lower() == 'github':
       await msg.answer('<b>Github репозиторий находится по ссылке:</b> \
       https://github.com/StanislaVolchkov/Neural_Style_Transfer_with_tg_bot ', parse_mode='HTML', reply_markup=kb)

   elif msg.text.lower() == 'примеры':
       await msg.answer('*Стилизация в зависимости от степени переноса:*', reply_markup=kb)
       await bot.send_photo(msg.chat.id, photo=types.InputFile("images/example.png"), reply_markup=kb)

   elif msg.text.lower() == '/2 в 1\U0001F315':
       await msg.answer('Раздел находится в разработке', reply_markup=kb)
       await msg.answer_sticker(r'CAACAgIAAxkBAAED-QJiEZra4fJTRPiscaZNmVyf-ayREQACFxMAAn0VkEqusV__xBuxTSME')
   else:
       await msg.answer('Не понимаю, что это значит.', reply_markup=kb)
       await msg.answer_sticker(r'CAACAgIAAxkBAAED-P5iEZozpTqbgFVW5Oxpip64N6mSowACDREAAs3amEoI7VMGrByieyME')

# Загрузка готовых стилей
async def get_prepared_style(msg: types.Message):
  if msg.text.lower() == 'готовые стили':
    await msg.answer('Можем предложить вам следующие варианты:', reply_markup=kb3)
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()

    media.attach_photo(types.InputFile("images/style/крик.jpg"), 'Эдвард Мунк "Крик"')
    media.attach_photo(types.InputFile("images/style/Женщина_с_шляпой.jpg"), 'Фанри Матис "Женщина с шляпой"')
    media.attach_photo(types.InputFile("images/style/Эшвилл.jpg"), 'Виллем де Кунинг "Эшвилл"')
    media.attach_photo(types.InputFile("images/style/mondrian.jpg"), 'Пит Модриан "Композиция в коричневом и сером"')
    media.attach_photo(types.InputFile("images/style/pencil_sketch.jpg"), 'Набросок женщины карандашом')
    await msg.answer_media_group(media)

  else:
    state = dp.get_current().current_state()
    if msg.text == 'Эдвард Мунк "Крик"':
      await state.update_data(style= "images/style/крик.jpg")
    elif msg.text == 'Фанри Матис "Женщина с шляпой"':
      await state.update_data(style= "images/style/Женщина_с_шляпой.jpg")
    elif msg.text == 'Виллем де Кунинг "Эшвилл"':
      await state.update_data(style= "images/style/Эшвилл.jpg")
    elif msg.text == 'Пит Модриан "Композиция в коричневом и сером"':
      await state.update_data(style= "images/style/mondrian.jpg")
    elif msg.text == 'Набросок женщины карандашом':
      await state.update_data(style= "images/style/pencil_sketch.jpg")

    await msg.answer(f'\U00002705Вы выбрали: *{msg.text}*')
    await FSMAdmin.next()
    await msg.answer('Введите сколько % стиля вы хотите перенести (от 0 до 100)')

# Результат работы модели
async def getting_model_output(msg, state):
  await msg.answer(f"Все будет в лучшем виде через пару секунд!\U0001F44C")
  await msg.answer_sticker(r'CAACAgIAAxkBAAED-QABYhGaUOcWpoUeaMAneSWcpL0da4MAApwTAALu0IhKTrrd85tUU3cjBA')
  async with state.proxy() as data:
    result = nst_model.get_transfer(data['content'], data['style'], data['percent'], 'weights.pth')
  await bot.send_photo(msg.chat.id, photo=result, reply_markup=kb)

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
  await msg.answer('Теперь нужно загрузить изображение, стиль которого хотите перенести! \
Или выберите из "Готовые стили"', reply_markup=kb2)

async def get_style_to_transfer(msg: types.Message, state=FSMContext):
  if msg.content_type == 'text':
    await get_prepared_style(msg)
  else:
    async with state.proxy() as data:
      logging.info(f"New User! Current number of users in dict: {msg}")
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
  dp.register_message_handler(start_transfer, commands=['Стилизация\U0001F311'], state=None)
  dp.register_message_handler(get_content_to_transfer, content_types=['photo'], state=FSMAdmin.content)
  dp.register_message_handler(get_style_to_transfer, content_types=['text', 'photo'], state=FSMAdmin.style)
  dp.register_message_handler(get_percent_to_transfer, content_types=['text'], state=FSMAdmin.percent)
  dp.register_message_handler(get_help_info, content_types=['text'])
  dp.register_message_handler(get_prepared_style, content_types=['text'])
  
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
        WEBAPP_PORT = os.environ['PORT']
        WEBAPP_HOST = '0.0.0.0'
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

