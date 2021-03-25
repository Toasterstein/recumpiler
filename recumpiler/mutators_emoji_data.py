import random

import emoji_data_python


def get_emoji_from_data(token: str):
    if len(token) < 4:
        return None
    else:
        emojis = [emoji.char for emoji in emoji_data_python.find_by_name(token)]
        if emojis:
            return random.choice(emojis)
        else:
            return None
