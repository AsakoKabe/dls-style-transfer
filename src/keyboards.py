from telebot.types import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton


def create_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard: ReplyKeyboardMarkup = ReplyKeyboardMarkup(True)
    keyboard.row('Сменить режим')
    keyboard.row('Настройки (Style Transfer)')
    keyboard.row('/help')

    return keyboard


def create_img_size_keyboard() -> InlineKeyboardMarkup:
    markup_img_size: InlineKeyboardMarkup = InlineKeyboardMarkup()

    markup_img_size.add(InlineKeyboardButton("128x128",
                                             callback_data='img_size_128x128'))
    markup_img_size.add(InlineKeyboardButton("256x256",
                                             callback_data='img_size_256x256'))
    markup_img_size.add(InlineKeyboardButton("280х310",
                                             callback_data='img_size_280x310'))
    markup_img_size.add(InlineKeyboardButton("320х250",
                                             callback_data='img_size_320x250'))
    markup_img_size.add(InlineKeyboardButton("728х420",
                                             callback_data='img_size_728x420'))
    markup_img_size.add(InlineKeyboardButton("1024х728",
                                             callback_data='img_size_1024x728'))

    return markup_img_size


def create_settings_keyboard() -> InlineKeyboardMarkup:
    markup_settings: InlineKeyboardMarkup = InlineKeyboardMarkup(row_width=2)

    markup_settings.add(InlineKeyboardButton("Изменить изображение стиля(Style Transfer)",
                                             callback_data='settings_img_style'))
    markup_settings.add(InlineKeyboardButton("Изменить размер обработанного изображения(Style Transfer)",
                                             callback_data='settings_img_size'))

    return markup_settings
