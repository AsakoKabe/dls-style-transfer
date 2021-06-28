import os
from pathlib import Path
from typing import Tuple

from telebot import TeleBot
from telebot.types import ReplyKeyboardMarkup, InlineKeyboardMarkup, File, Message
from torchvision import transforms
from torch import Tensor
from PIL import Image

from keyboards import create_main_keyboard, create_img_size_keyboard, create_settings_keyboard
from style_transfer.models import StyleTransfer
from style_transfer.utils import save_img
from AnimeColorDeOldify.deoldify.visualize import get_image_colorizer, ModelImageVisualizer

token: str = os.environ.get('BOT_TOKEN')
bot: TeleBot = TeleBot(token)
main_keyboard: ReplyKeyboardMarkup = create_main_keyboard()
settings_keyboard: InlineKeyboardMarkup = create_settings_keyboard()
change_img_size_keyboard: InlineKeyboardMarkup = create_img_size_keyboard()

style_transfer_model: StyleTransfer = StyleTransfer()
colorizer: ModelImageVisualizer = get_image_colorizer(artistic=True, stats=([0.7137, 0.6628, 0.6519], [0.2970, 0.3017, 0.2979]))


@bot.message_handler(commands=['start'])
def send_start_message(message):
    bot.send_message(message.chat.id, 'Бот для переноса стиля изображений. \n'
                                      'Проект выполнен в рамках курса DLS 1 семестра. \n'
                                      'Доступно 2 режима для работы: \n'
                                      '1) Перенос стиля одного изображения на другое (Style Transfer)\n'
                                      '2) "Раскрашивание" grayscale изображений (DeOldify)\n'
                                      'Для получение справки о том, как пользовать ботом нажми /help". \n',
                     reply_markup=main_keyboard)


@bot.message_handler(commands=['help'])
def send_help_message(message):
    bot.send_message(message.chat.id, 'По умолчанию включен режим Style Transfer и установлен стиль Picasso. \n'
                                      'Чтобы установить свой стиль нужно перейти в настройки, нажить '
                                      '"Изменить изображение стиля(Style Transfer) " \n'
                                      'и загрузить новое изображение. \n'
                                      'Для получившихся изображений нужно установить размер изображения. По умолчанию '
                                      'установлено 128x128. \nОт размера изображения зависит скорость обработки. Чем '
                                      'больше размер изображение, тем дольше будет обрабатываться входное '
                                      'изображение. \n'
                                      '\n'
                                      'Для второго режима (DeOldify) нужно переключиться на него с помощью '
                                      'кнопки "Сменить режим". Далее отправлять свои изображения. \n'
                                      '\n'
                                      'Ссылка на репозиторий: https://github.com/AsakoKabe/dls-style-transfer\n'
                                      'Ссылка на репозиторий DeOldify: https://github.com/Dakini/AnimeColorDeOldify',
                     reply_markup=main_keyboard)


@bot.message_handler(content_types=['text'])
def text_parse(message):
    if message.text == 'Настройки (Style Transfer)':
        bot.send_message(message.chat.id, 'Настройки:', reply_markup=settings_keyboard)
    elif message.text == 'Сменить режим':
        style_transfer_model.change_mode()
        if style_transfer_model.get_mode():
            bot.send_message(message.chat.id, 'Выбран (Style Transfer)')
        else:
            bot.send_message(message.chat.id, 'Выбран (DeOldify)')


def get_path_image_from_message(message: Message) -> str:
    file_info: File = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file: bytes = bot.download_file(file_info.file_path)
    path: str = 'images/' + message.photo[1].file_id
    with open(path, 'wb') as new_file:
        new_file.write(downloaded_file)

    return path


def send_img_and_clear_output(message: Message, msg: Message, path_image_input: str, path_image_output: str) -> None:
    bot.delete_message(message.chat.id, msg.message_id)
    bot.send_photo(message.chat.id, photo=open(path_image_output, 'rb'))
    if path_image_output != 'images/picasso.jpg':
        os.remove(path_image_output)
    os.remove(path_image_input)


@bot.message_handler(content_types=['photo'])
def image_processing(message):
    bot.send_chat_action(message.chat.id, action='upload_photo')

    path_image_input: str = get_path_image_from_message(message)
    path_image_output: str = 'images/' + str(message.chat.id) + '_output.jpg'

    msg: Message = bot.send_message(message.chat.id, 'Обрабатываем...')
    if style_transfer_model.get_mode():
        style_transfer_model.set_content_img(path_image_input)
        output_img: Tensor = style_transfer_model.fit()
        save_img(output_img, path_image_output)
        send_img_and_clear_output(message, msg, path_image_input, path_image_output)
    else:
        tensor_to_pil: transforms.ToTensor = transforms.ToTensor()
        output_img_pil: Image = colorizer.get_transformed_image(path=Path(path_image_input), render_factor=30)
        output_img_tensor: Tensor = tensor_to_pil(output_img_pil)
        save_img(output_img_tensor, path_image_output)
        send_img_and_clear_output(message, msg, path_image_input, path_image_output)


@bot.callback_query_handler(func=lambda call: call.data.startswith('img_size_'))
def callback_change_img_size(call):
    # TODO: Добавить вариант с размерами входного изображения
    new_img_size: Tuple[int] = tuple(map(int, call.data.split('_')[-1].split('x')))
    style_transfer_model.set_img_size(new_img_size)
    if style_transfer_model.get_style_img() is not None:
        style_transfer_model.set_style_img(style_transfer_model.get_style_img_path())
    bot.send_message(call.message.chat.id, 'Сохранено')


def set_style_image_reply(message):
    try:
        style_image_path: str = get_path_image_from_message(message)
        style_transfer_model.set_style_img(style_image_path)
        old_style_image_path: str = style_transfer_model.get_style_img_path()
        if old_style_image_path != 'images/picasso.jpg':
            os.remove(old_style_image_path)
        style_transfer_model.set_style_img_path(style_image_path)
        bot.send_message(message.chat.id, "Изображение сохранено")
    except TypeError:
        bot.send_message(message.chat.id, "Ошибка при сохранении изображения. (Это точно изображение?)")


@bot.callback_query_handler(func=lambda call: call.data == 'settings_img_style')
def callback_settings_img_style(call):
    message: Message = bot.send_message(call.message.chat.id, "Отправь изображение стиля")
    bot.register_next_step_handler(message, set_style_image_reply)


@bot.callback_query_handler(func=lambda call: call.data == 'settings_img_size')
def callback_settings_img_size(call):
    bot.send_message(call.message.chat.id, 'Возможные размеры изображения:', reply_markup=change_img_size_keyboard)


if __name__ == '__main__':
    bot.polling()
