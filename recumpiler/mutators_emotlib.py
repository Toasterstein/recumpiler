import random

import emotlib


def get_emoticon(token: str):
    # TODO: use WordNEt with NTLK to find works similar to the emoticon keys?
    # TODO: maybe use sentiment analysis
    emoticons = emotlib.EMOTICON_UNICODE.get(token, None)
    if emoticons:
        return random.choice(emoticons)
    else:
        return None
