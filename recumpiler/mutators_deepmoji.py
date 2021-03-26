# pylint: disable=import-error

# TODO: pylint disables are a byproduct of us unable to install
#       deepmoji in CI
import random
from typing import List

import numpy as np

from deepmoji.global_variables import PRETRAINED_PATH, get_vocabulary
from deepmoji.model_def import deepmoji_emojis
from deepmoji.sentence_tokenizer import SentenceTokenizer


import tensorflow as tf

# from keras.backend.tensorflow_backend import set_session

# gpu compatibility for TF 2.0 with legacy keras
config = tf.compat.v1.ConfigProto()
# pylint: disable=no-member
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = (
    True  # to log device placement (on which device the operation ran)
)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
# set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)

EMOJI_MAP = {
    0: "😂",
    1: "😒",
    2: "😩",
    3: "😭",
    4: "😍",
    5: "😔",
    6: "👌",
    7: "😊",
    8: "❤",
    9: "😏",
    10: "😁",
    11: "🎶",
    12: "😳",
    13: "💯",
    14: "😴",
    15: "😌",
    16: "☺",
    17: "🙌",
    18: "💕",
    19: "😑",
    20: "😅",
    21: "🙏",
    22: "😕",
    23: "😘",
    24: "♥",
    25: "😐",
    26: "💁",
    27: "😞",
    28: "🙈",
    29: "😫",
    30: "✌",
    31: "😎",
    32: "😡",
    33: "👍",
    34: "😢",
    35: "😪",
    36: "😋",
    37: "😤",
    38: "✋",
    39: "😷",
    40: "👏",
    41: "👀",
    42: "🔫",
    43: "😣",
    44: "😈",
    45: "😓",
    46: "💔",
    47: "♡",
    48: "🎧",
    49: "🙊",
    50: "😉",
    51: "💀",
    52: "😖",
    53: "😄",
    54: "😜",
    55: "😠",
    56: "🙅",
    57: "💪",
    58: "👊",
    59: "💜",
    60: "💖",
    61: "💙",
    62: "😬",
    63: "✨",
}


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def elements_past_min(array, min_dist: float):
    return np.argwhere(array > min_dist).flatten()


def get_top_n_emojis(
    st, deepmoji_model, sentence: str, most_n: int = 5, min_dist: float = None
) -> List[str]:
    tokenized, _, _ = st.tokenize_sentences([sentence])
    prob = deepmoji_model.predict(tokenized)
    for i, t_prob in enumerate(prob):
        if min_dist is not None:
            ids = list(
                i
                for i in top_elements(t_prob, most_n)
                if i in elements_past_min(t_prob, min_dist)
            )
        else:
            ids = list(top_elements(t_prob, most_n))
        return list([EMOJI_MAP[emoji_index] for emoji_index in ids])


sentence_tokenizer = SentenceTokenizer(get_vocabulary(), 30)
deepmoji_model = deepmoji_emojis(
    maxlen=30,
    weight_path=PRETRAINED_PATH,
)


deepmoji_model.summary()


def sentiment_query(word: str, most_n: int = 5, min_dist: float = None):
    return get_top_n_emojis(
        sentence_tokenizer, deepmoji_model, word, most_n=most_n, min_dist=min_dist
    )


sentiment_query("I lost my dog oh no")


import enum


class deepmoji_lookup_accuracy(enum.Enum):
    low = 0.01
    med = 0.04
    high = 0.07


def get_sentiment_emoji(sentance):
    emojis = sentiment_query(
        str(sentance), most_n=3, min_dist=deepmoji_lookup_accuracy.high.value
    )
    if emojis:
        # TODO: maybe smarter grab
        return random.choice(emojis)
    return None
