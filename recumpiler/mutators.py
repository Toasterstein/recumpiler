#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""garbage code to make garbage text"""

import random
import re
import string
from functools import wraps
from math import ceil
from typing import List, Optional
from logging import getLogger
from timeit import default_timer as timer

import homoglyphs as hg
import inflect
import nltk
import numpy as np
import pronouncing
from better_profanity import profanity
from nltk.corpus import wordnet as wn
from nltk.tokenize.casual import (
    HANG_RE,
    reduce_lengthening,
    remove_handles,
    _replace_html_entities,
    WORD_RE,
    EMOTICON_RE,
    REGEXPS,
)
from nltk.tokenize.treebank import TreebankWordDetokenizer
from regex import regex
from textblob import TextBlob, Word, Sentence
from textblob.base import BaseTokenizer
from textblob.utils import strip_punc
from word2number import w2n

# TODO: issues with pyenchant
# import splitter
from recumpiler.cheap_emoji_alias import get_cheap_emoji_alias

# these are imported like this as the dependencies are complicated to install
# and require large files. These are annoying to test in a CI for now.
# TODO: avoid this work around in future make better CI solution
try:
    from recumpiler.emojijnet import get_gloveword_emoji
except:
    get_gloveword_emoji = lambda word: None
try:
    from recumpiler.mutators_deepmoji import get_sentiment_emoji
except:
    get_sentiment_emoji = lambda sentence: None


from recumpiler.mutators_emoji_data import get_emoji_from_data
from recumpiler.mutators_emotlib import get_emoticon
from recumpiler.utils import (
    load_simple_text_emojis,
    load_action_verbs,
    load_rp_pronouns,
    init_emoji_database,
    get_emoji_database,
    load_text_face_emoji,
    load_garbage_tokens,
    decision,
)

inflect_engine = inflect.engine()

__log__ = getLogger(__name__)


def logged_mutator(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = timer()
        output = f(*args, **kwds)
        end = timer()
        # TODO: issue hitting recursion limit
        # __log__.info(
        #     {
        #         "message": "called mutator",
        #         "mutator": f.__name__,
        #         "args": args,
        #         "kwargs": kwds,
        #         "output": output,
        #         "exc_time": "{0:.15f}".format(end - start),
        #     }
        # )
        return output

    return wrapper


# TODO: refactor this global garbage


num_to_word_probability = 0.3
word_to_num_probability = 0.3

common_misspellings_probability = 0.2
hard_owo_replace_probability = 0.2

bold_text_probability = 0.04

REEE_probability = 0.06
REEE_allcaps_probability = 0.3

add_random_rp_action = True
add_random_rp_mid_sentence_action_probability = 0.005
add_random_rp_end_sentence_action_probability = 0.02
more_verbs_probability_decay = 0.4

add_random_garbage = True
add_random_garbage_probability = 0.01

add_random_plurals = True
add_random_plurals_probability = 0.1

randomly_lemmatize = True
randomly_lemmatize_probability = 0.1

randomly_overemphasis_punctuation = True
randomly_overemphasis_punctuation_probability = 0.5
randomly_overemphasis_punctuation_max_fuck = 4

randomly_capitalize_word = True
randomly_capitalize_word_probability = 0.1

randomly_spongebob_word = True
randomly_spongebob_word_probability = 0.1

add_randomly_text_face_emoji = True
add_randomly_text_face_emoji_probability = 0.05

add_random_simple_text_emoji = True
add_random_simple_text_emoji_probability = 0.07

randomly_swap_char = True
randomly_swap_char_probability = 0.04
randomly_swap_char_swap_percent = 0.2

randomly_insert_char = True
randomly_insert_char_probability = 0.04
randomly_insert_char_insert_percent = 0.1

random_leet_speak = True
random_leet_speak_probability = 0.1

utf_8_char_swaps_probability = 0.1

random_censor_probability = 0.01
random_censor_percent = 0.25

censor_profanity_probability = 0.7
censor_profanity_percent = 0.25

random_synonym_probability = 0.5

random_ending_y_probability = 0.05
leet_speak_min_token_length = 5

adding_ending_ksksk_andioop_probability = 0.8
adding_ending_ksksk_save_the_turtles_probability = 0.3

ksksk_enlargement_probability = 0.7

owo_vs_ouo_bias = 0.5

add_extra_ed_probability = 0.05
split_compound_word_probability = 0.03

lazy_char_subbing_probability = 0.6
uck_to_ucc_swap_probability = 0.4

juwuice_swap_probability = 0.5

add_x3_if_token_has_rawr_probability = 0.2
me_2_meh_swap_probability = 0.5
me_2_meow_swap_probability = 0.5

hard_uwu_replace_probability = 0.3
sub_to_subby_swap_probability = 0.3

fucking_normies_addition = 0.3

get_rhymes_probability = 0.01
max_runon_rhymes = 3

homofiy_probability = 0.3
homofiy_percentage = 0.3

back_tick_text_probability = 0.05

space_gap_text_probability = 0.02
space_gap_text_min_gap_size = 1
space_gap_text_max_gap_size = 4

add_text_relevant_emoji_probability = 0.1
wrap_text_relevant_emoji_probability = 0.02

lr_to_w_swap_probability = 0.4


@logged_mutator
def num_to_word(token: str) -> str:
    try:
        return str(w2n.word_to_num(token))
    except ValueError:
        return token


@logged_mutator
def word_to_num(token: str) -> str:
    try:
        return inflect_engine.number_to_words(int(token))
    except ValueError:
        return token


@logged_mutator
def knotter(token: str) -> str:
    token = re.sub(
        r"(([^kK]|^)no+t)",
        lambda match: f"kn{'o' * random.choice(range(1, 3))}t",
        token,
        flags=re.IGNORECASE,
    )
    return token


@logged_mutator
def homoify(token: str, homo_percent: float = 0.3):
    if len(token) <= 3:  # dont homoglyph censor stuff this small
        return token
    swaps = int(ceil(len(token) * homo_percent))
    indexes = random.choices(range(1, len(token)), k=swaps)
    for i in indexes:
        token = "".join(
            [
                token[w]
                if w != i
                else random.choice(hg.Homoglyphs().get_combinations(token[w]))
                for w in range(len(token))
            ]
        )
    return token


@logged_mutator
def owoer(token: str) -> str:
    # TODO: owo usually goes to owoo should supress.

    token = re.sub(
        r"(ou)([^o]|$)",
        lambda match: f"ouo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )
    token = re.sub(
        r"(ow)([^o]|$)",
        lambda match: f"owo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )
    token = re.sub(
        r"(ov)([^o]|$)",
        lambda match: f"ovo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )

    token = re.sub(r"(cor)", lambda match: f"cowor", token)

    if (
        "owo" not in token.lower()
        and "ouo" not in token.lower()
        and decision(hard_owo_replace_probability)
    ):
        owo_str = "owo" if decision(owo_vs_ouo_bias) else "ouo"
        token = re.sub(
            r"(o+)",
            lambda match: (owo_str * len(match.group(1))).replace("oo", "o"),
            token,
            flags=re.IGNORECASE,
            count=random.choice(range(0, 2)),
        )

    # TODO: UWU
    # juice -> juwuice
    if decision(juwuice_swap_probability):
        token = re.sub(
            r"u+(i?ce)",
            lambda match: f"uwu{match.group(1)}",
            token,
            flags=re.IGNORECASE,
        )

    if "uwu" not in token.lower() and decision(hard_uwu_replace_probability):
        uwu_str = "uwu"
        token = re.sub(
            r"u+", uwu_str, token, flags=re.IGNORECASE, count=random.choice(range(0, 2))
        )

    return token


@logged_mutator
def fuckyer(token: str) -> str:
    extra_fun = ""
    y_choice_1 = ("y" if decision(0.5) else "i") * random.choice(range(1, 5))
    y_choice_2 = ("y" if decision(0.5) else "i") * random.choice(range(1, 5))
    if decision(0.5):
        extra_fun = f"w{'u' * random.choice(range(1, 5))}k{y_choice_2}"
    token = re.sub(
        r"([Ff])?uck(er|ing)?",
        lambda match: f"{match.group(1) or ''}{'u' * random.choice(range(1,5))}k{y_choice_1}{match.group(2) or ''}"
        + " "
        + extra_fun,
        token,
    )
    return token


@logged_mutator
def garbage(token: str) -> str:
    # inserting gay
    token = re.sub(r"([a-fh-zA-FH-Z])a+y+", lambda match: f"{match.group(1)}gay", token)

    # hello -> hewwo
    token = re.sub(r"([Hh])e+ll+o+?", lambda match: f"{match.group(1)}ewwo", token)

    # er -> ur
    if decision(0.4):
        token = re.sub(
            r"e+r+",
            lambda match: f"u{'r' * ceil(np.random.rayleigh(1.2))}",
            token,
            flags=re.IGNORECASE,
        )

    #  ello - >ewwo
    if decision(0.4):
        token = re.sub(
            r"e+ll+o+?",
            lambda match: f"ew{'w' * ceil(np.random.rayleigh(1.2))}o",
            token,
            flags=re.IGNORECASE,
        )  # 2-6ish

    # cute -> koot
    token = re.sub(
        r"([Cc])u+te",
        lambda match: f"{match.group(1)}oo{'o' * random.randint(0,5)}t",
        token,
    )

    # ove -> wuv
    if decision(0.7):
        token = re.sub(r"(o+)ve", lambda match: f"w{'u' * len(match.group(1))}v", token)

    # one -> wun
    if decision(0.7):
        token = re.sub(r"one", "wun", token, flags=re.IGNORECASE)

    # as -> ass asss
    if decision(0.5):
        token = re.sub(
            r"([aA])([sS])($|[^s])",
            lambda match: f"{match.group(1)}{match.group(2) * random.randint(2,3)}t",
            token,
        )

    # TODO: refactor (me -> meh|me -> meow) together?
    # me -> meow
    if decision(me_2_meow_swap_probability):
        token = re.sub(
            r"^me+$",
            lambda match: f"m{'e' * random.randint(1,3)}{'o' * random.randint(1,3)}w",
            token,
            flags=re.IGNORECASE,
        )

    # me -> meh
    if decision(me_2_meh_swap_probability):
        token = re.sub(
            r"^me+$",
            lambda match: f"m{'e' * random.randint(1, 3)}h",
            token,
            flags=re.IGNORECASE,
        )

    # my -> mah, myah
    if decision(0.5):
        token = re.sub(
            r"^my+$",
            lambda match: f"m{'y' if decision(0.3) else ''}{'a' * random.randint(2, 3)}{'h' if decision(0.5) else ''}",
            token,
        )

    # ion -> shun
    if decision(0.5):
        token = re.sub(r"ion$", "shun", token)

    # .ome -> .um
    if decision(0.5):
        token = re.sub(r"([a-zA-Z])ome", lambda match: f"{match.group(1)}um", token)

    # teh or da
    if decision(0.5):
        token = re.sub(r"^([Tt])he$", lambda match: f"{match.group(1)}eh", token)
    else:
        token = re.sub(
            r"^([Tt])he$",
            lambda match: f"{'D' if match.group(1) == 'T' else 'd'}a",
            token,
        )

    # ing -> inn
    if decision(0.5):
        token = re.sub(
            r"ing$",
            f"in{'n' * random.randint(0,4) if decision(0.5) else 'in' * random.randint(0, 4)}",
            token,
            flags=re.IGNORECASE,
        )

    # ks -> ksksksk
    if decision(ksksk_enlargement_probability):
        token = re.sub(
            r"[kK][sS]|[sS][kK]",
            lambda match: f"{match.group(0) * random.randint(2,6)}",
            token,
            flags=re.IGNORECASE,
        )

    # uck -> ucc, uccci
    if decision(uck_to_ucc_swap_probability):
        token = re.sub(
            r"u+c+k+",
            lambda match: f"u{'c' * random.randint(2,6)}{'i' * random.randint(0,3)}",
            token,
            flags=re.IGNORECASE,
        )

    if decision(sub_to_subby_swap_probability):
        token = re.sub(
            r"s(u+)b",
            lambda match: f"s{match.group(1)}bb{('y' if decision(0.5) else 'i') * random.randint(1, 2)}",
            token,
            flags=re.IGNORECASE,
        )

    # no -> nu+ nyu+
    if decision(0.5):
        token = re.sub(
            "([nN])(o+)",
            lambda match: f"{match.group(1)}{'y' if decision(0.5) else ''}{'u' * (len(match.group(2)) * random.randint(1, 6))}",
            token,
            flags=re.IGNORECASE,
        )
    return token


@logged_mutator
def reeeer(token: str) -> str:
    if decision(REEE_probability):
        token = re.sub(
            r"([Rr])e*",
            lambda match: f"{match.group(1)}e" + "e" * random.choice(range(1, 15)),
            token,
        )
        if decision(REEE_allcaps_probability):
            token = token.upper()
    return token


def rawrer(token: str) -> str:
    token = re.sub(r"ra([a-zA-Z])?", lambda match: f"rawr{match.group(1) or ''}", token)
    token = re.sub(
        r"ar([a-zA-Z])?", lambda match: f"arawr{match.group(1) or ''}", token
    )
    token = re.sub(r"([Rr])oar", lambda match: f"{match.group(1)}awr", token)

    return token


@logged_mutator
def lr_to_w_swap(token: str) -> str:
    token = re.sub(
        r"([lL])",
        lambda match: f"{('w' if decision(0.7) else 'wl') if match.group(1).islower() else ('W' if decision(0.7) else 'WL')}",
        token,
    )
    token = re.sub(
        r"([rR])",
        lambda match: f"{('w' if decision(0.7) else 'wr') if match.group(1).islower() else ('W' if decision(0.7) else 'WR')}",
        token,
    )
    return token


@logged_mutator
def jizzer(token: str) -> str:
    token = re.sub(r"(.iz+)", "jizz", token)
    return token


# TODO: reval this some of it is ineffective i think
@logged_mutator
def cummer(token: str) -> str:
    token = re.sub(r"(.ome|co+m|co+n{1,3})", "cum", token)
    token = re.sub(r"(c.{0,2}u+m)", "cum", token)
    token = re.sub(r"(cau|cou)", "cum", token)
    token = re.sub(r"(cow)", "cum", token)
    token = re.sub(r"(son|sun$)", "cum", token)

    token = re.sub(r"([a-bd-zA-BD-Z])um", lambda match: f"{match.group(1)}cum", token)

    token = re.sub(
        r"([a-bd-zA-BD-Z])u(nn|mm)([yi])",
        lambda match: f"{match.group(1)}cumm{match.group(3)}",
        token,
    )
    token = re.sub(r"(cally)", "cummy", token)

    return token


garbage_tokens = load_garbage_tokens()


@logged_mutator
def add_random_garbage_token():
    return random.choice(garbage_tokens)


text_face_emojis = load_text_face_emoji()


@logged_mutator
def find_text_relevant_emoji(token: str) -> Optional[str]:
    if (
        len(token) < 4
    ):  # TODO: find better logic to avoid getting garbage or complete unrelated emojis
        return
    results = (
        get_emoji_database()
        .execute(
            """select Emoji from Emoji_Sentiment_Data where "Unicode name" LIKE ?""",
            ("%" + token.upper() + "%",),
        )
        .fetchall()
    )
    if results:
        return random.choice(results)[0]


emoji_database = init_emoji_database()

simple_text_emojis = load_simple_text_emojis()

action_verbs = load_action_verbs()

rp_pronouns = load_rp_pronouns()


@logged_mutator
def get_random_text_face_emojis():
    return random.choice(text_face_emojis)


@logged_mutator
def get_random_simple_text_emojis():
    return random.choice(simple_text_emojis)


@logged_mutator
def generate_spongebob_text(token: str) -> str:
    """gEnErAtEs sPoNgEbOb mEmE TeXt"""
    spongebob_text = ""
    for i, char in enumerate(token):
        if i % 2 == 0:
            spongebob_text += char.lower()
        else:
            spongebob_text += char.upper()
    return spongebob_text


@logged_mutator
def shuffle_str(token: str) -> str:
    token_str_list = list(token)
    random.shuffle(token_str_list)
    return "".join(token_str_list)


@logged_mutator
def get_runon_of_rhymes(
    token: str,
    max_runon: int = 3,
    allow_token_dupe: bool = False,
    allow_rhyme_dupes: bool = False,
) -> List[str]:
    # TODO: this is a complicated mess
    selected_rhymes = []

    rhymes = get_pronouncing_rhyme(token)
    if not allow_token_dupe:
        try:
            rhymes.remove(token)
        except ValueError:
            pass

    level = 4
    while True:
        rhymes += get_nltk_rymes(token, level)
        if not allow_token_dupe:
            try:
                rhymes.remove(token)
            except ValueError:
                pass
        if rhymes:
            break
        if level == 0 or len(rhymes) > max_runon:
            break
        level -= 1

    if not allow_token_dupe:
        try:
            rhymes.remove(token)
        except ValueError:
            pass
    if not allow_rhyme_dupes:
        rhymes = list(sorted(list(set(rhymes))))
    if rhymes:
        selected_rhymes += random.choices(rhymes, k=min(len(rhymes), max_runon))
    return selected_rhymes


@logged_mutator
def get_pronouncing_rhyme(token: str) -> List[str]:
    return pronouncing.rhymes(token)


@logged_mutator
def get_nltk_rymes(token: str, level) -> List[str]:
    # TODO: stub
    def rhyme(inp, level):
        """
        1 bad rhymes
        2
        4 good rhymes
        """
        entries = nltk.corpus.cmudict.entries()
        syllables = [(word, syl) for word, syl in entries if word == inp]
        rhymes = []
        for (word, syllable) in syllables:
            rhymes += [
                word for word, pron in entries if pron[-level:] == syllable[-level:]
            ]
        return set(rhymes)

    return list(rhyme(token, level))


@logged_mutator
def over_emphasise_punctuation(token: str, max_fuck: int = 4) -> str:
    if token == "?":
        token += "".join(
            random.choices(
                [
                    "1",
                    # "i",
                    "!",
                    "?",
                    # "I",
                    # "/",
                    # ".",
                    # "\\"
                ],
                k=random.choice(range(0, max_fuck)),
            )
        )
        token = shuffle_str(token)
    if token == "!":
        token += "".join(
            random.choices(
                [
                    "1",
                    # "i",
                    "!",
                    "?",
                    # "I",
                    # "/",
                    "|",
                ],
                k=random.choice(range(0, max_fuck)),
            )
        )
        token = shuffle_str(token)
    if token == ".":
        token += "".join(
            random.choices([",", "."], k=random.choice(range(0, max_fuck)))
        )
        token = shuffle_str(token)

    return token


@logged_mutator
def to_rp_text(token: str) -> str:
    return f"*{token}*"


@logged_mutator
def get_random_action_verb():
    return random.choice(action_verbs)


@logged_mutator
def get_random_rp_pronoun():
    return random.choice(rp_pronouns)


@logged_mutator
def random_swap_char(token: str, swaps_percent: float = 0.2) -> str:
    if len(token) < 3:  # dont do this for small tokens as they become un decipherable
        return token
    swaps = int(ceil(len(token) * swaps_percent))
    indexes = random.choices(range(len(token)), k=swaps)
    for i in indexes:
        token = "".join(
            [
                token[w] if w != i else random.choice(string.ascii_letters)
                for w in range(len(token))
            ]
        )
    return token


@logged_mutator
def random_insert_char(token: str, insert_percent: float = 0.1) -> str:
    swaps = int(ceil(len(token) * insert_percent))
    indexes = random.choices(range(len(token)), k=swaps)
    token_str_list = list(token)
    for i in indexes:
        token_str_list.insert(i, random.choice(string.ascii_letters))
    token = "".join(token_str_list)
    return token


@logged_mutator
def token_to_leet(token: str) -> str:
    if len(token) < 5:  # leet speaking small text has hard to read results
        return token
    leet_char_mapping = {
        # "a": "4",
        "a": "@",
        "e": "3",
        "8": "&",
        "l": "1",
        "o": "0",
        "s": "5",
        "i": "1",
    }
    getchar = (
        lambda c: leet_char_mapping[c.lower()] if c.lower() in leet_char_mapping else c
    )
    return "".join(getchar(c) for c in token)


# TODO: lots of options maybe something learned?
@logged_mutator
def utf_8_char_swaps(token: str) -> str:
    if decision(0.5):
        token = re.sub(r"ae", "æ", token)
        token = re.sub(r"AE", "Æ", token)
    if decision(0.3):
        token = re.sub(r"ea", "æ", token)
        token = re.sub(r"EA", "Æ", token)
    return token


# TODO: this is only for discord so we don't break tokenization
class TweetTokenizer:
    r"""
    Tokenizer for tweets.

        >>> from nltk.tokenize import TweetTokenizer
        >>> tknzr = TweetTokenizer()
        >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
        >>> tknzr.tokenize(s0)
        ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    Examples using `strip_handles` and `reduce_len parameters`:

        >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
        >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
        >>> tknzr.tokenize(s1)
        [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    """

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        :param text: str
        :rtype: list(str)
        :return: a tokenized list of strings; concatenating this list returns\
        the original string if `preserve_case=False`
        """
        # Fix HTML character entities:
        text = _replace_html_entities(text)
        # Remove username handles
        if self.strip_handles:
            text = remove_handles(text)
        # Normalize word lengthening
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Shorten problematic sequences of characters
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # Tokenize:
        r"|<(?:[^\d>]+|:[A-Za-z0-9]+:)\w+>"
        custom_Re = regex.compile(
            r"""(%s)"""
            % "|".join(
                (
                    r":[^:\s]+:",
                    r"<:[^:\s]+:[0-9]+>",
                    r"<a:[^:\s]+:[0-9]+>",
                    r"<(?:[^\d>]+|:[A-Za-z0-9]+:)\w+>",
                )
                + REGEXPS
            ),
            regex.VERBOSE | regex.I | regex.UNICODE,
        )
        words = custom_Re.findall(safe_text)
        # Possibly alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = list(
                map((lambda x: x if EMOTICON_RE.search(x) else x.lower()), words)
            )
        return words


class TweetWordTokenizer(BaseTokenizer):
    """NLTK's recommended word tokenizer (currently the TreeBankTokenizer).
    Uses regular expressions to tokenize text. Assumes text has already been
    segmented into sentences.

    Performs the following steps:

    * split standard contractions, e.g. don't -> do n't
    * split commas and single quotes
    * separate periods that appear at the end of line
    """

    def tokenize(self, text, include_punc=True):
        """Return a list of word tokens.

        :param text: string of text.
        :param include_punc: (optional) whether to include punctuation as separate tokens. Default to True.
        """
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = TweetTokenizer().tokenize(text)
        if include_punc:
            return tokens
        else:
            # Return each word token
            # Strips punctuation unless the word comes from a contraction
            # e.g. "Let's" => ["Let", "'s"]
            # e.g. "Can't" => ["Ca", "n't"]
            # e.g. "home." => ['home']
            return [
                word if word.startswith("'") else strip_punc(word, all=False)
                for word in tokens
                if strip_punc(word, all=False)
            ]


@logged_mutator
def recumpile_sentence(sentence: Sentence) -> List[str]:
    new_tokens = []
    # TODO: determine mood classifier for sentence and add respective emoji
    sentiment_emoji = None
    if decision(0.89):
        sentiment_emoji = get_sentiment_emoji(sentence)

    for token in sentence.tokenize(TweetWordTokenizer()):
        # TODO: this is only for discord so we dont break tokenization
        if re.match(
            r"@everyone|@here|<:[^:\s]+:[0-9]+>|<a:[^:\s]+:[0-9]+>|<(?:@!?\d+|:[A-Za-z0-9]+:)\w+>",
            token,
        ):
            new_tokens.append(token)
            continue

        emoji = None
        alias_emoji = get_cheap_emoji_alias(token)

        # TODO: refactor into its own mutator
        if decision(0.9) and (
            re.match("among", token, flags=re.IGNORECASE)
            or re.match("amogus", token, flags=re.IGNORECASE)
            or re.match(r"su+s", token, flags=re.IGNORECASE)
        ):
            emoji = "ඞ"

        emoticon = get_emoticon(token)

        if alias_emoji:
            if decision(0.1) or (len(str(token)) == 1 and decision(0.9)):
                new_tokens.append(alias_emoji)
                continue
            else:
                if decision(0.5):
                    new_tokens.append(alias_emoji)

        if decision(0.5):
            emoji = get_emoji_from_data(token)
        if decision(0.3):
            emoji = get_gloveword_emoji(token)
        if emoji:
            if decision(0.5):
                new_tokens.append(emoji)

        if decision(random_synonym_probability):
            token = replace_with_random_synonym(token)
        if decision(0.5) and profanity.contains_profanity(token):
            token = token.upper()
        if decision(censor_profanity_probability) and profanity.contains_profanity(
            token
        ):
            if decision(0.1):
                token = custom_censoring(token, 1)
            else:
                token = custom_censoring(token, censor_profanity_percent)
        elif decision(random_censor_probability):
            token = custom_censoring(token, random_censor_percent)

        if re.match("musk", token, flags=re.IGNORECASE):
            add_husky = True
        else:
            add_husky = False

        # processing
        recumpiled_token = recumpile_token(token)

        # post processing
        new_tokens.append(recumpiled_token)

        if emoji:
            if decision(0.8):
                new_tokens.append(emoji)
        if alias_emoji:
            if decision(0.8):
                new_tokens.append(alias_emoji)
        if emoticon:
            if decision(0.8):
                new_tokens.append(emoticon)

        if add_husky:
            new_tokens.append(recumpile_token("husky"))

        if add_random_garbage and decision(add_random_garbage_probability):
            new_tokens.append(recumpile_token(add_random_garbage_token()))
        if add_randomly_text_face_emoji and decision(
            add_randomly_text_face_emoji_probability
        ):
            new_tokens.append(get_random_text_face_emojis())
        if add_random_simple_text_emoji and decision(
            # TODO: use textblob to determine mood of text and insert faces
            #  accordingly likely need to do this after reconstruction of the
            #  text blob and go through this sentence by sentence rather than
            #  word by word.
            add_random_simple_text_emoji_probability
        ):
            new_tokens.append(get_random_simple_text_emojis())
        if add_random_rp_action and decision(
            add_random_rp_mid_sentence_action_probability
        ):
            new_tokens.append(get_random_rp_action_sentence())
    if add_random_rp_action and decision(add_random_rp_end_sentence_action_probability):
        new_tokens.append(get_random_rp_action_sentence())

    if sentiment_emoji:
        new_tokens.append(sentiment_emoji)
        if decision(0.4):
            for i in range(5):
                if decision(0.3):
                    new_tokens.append(sentiment_emoji)
                else:
                    break

    return new_tokens


@logged_mutator
def add_ending_y(token: str) -> str:
    return re.sub(r"([a-zA-Z]{4,}[^sy])", lambda match: f"{match.group(1)}y", token)


@logged_mutator
def recumpile_token(token: str) -> str:
    # TODO: determine mood classifier for token and add respective emoji
    if decision(split_compound_word_probability):
        tokens = split_compound_word(token)
    else:
        tokens = [token]

    # TODO: migrate fuck_token to maybe a generator?
    fucked_tokens = []
    for token in tokens:
        relevant_emoji = None
        if decision(add_text_relevant_emoji_probability):
            relevant_emoji = find_text_relevant_emoji(
                token
            )  # TODO: add ability to get multiple?
            if relevant_emoji and decision(wrap_text_relevant_emoji_probability):
                fucked_tokens.append(relevant_emoji)

        if decision(lazy_char_subbing_probability):
            token = lazy_char_subbing(token)

        # TODO: this is a potential for unexpected behavior
        if decision(word_to_num_probability):
            token = word_to_num(token)
        if decision(num_to_word_probability):
            token = num_to_word(token)

        if decision(lr_to_w_swap_probability):
            token = lr_to_w_swap(token)

        fucked_token = knotter(fuckyer(reeeer(rawrer(garbage(owoer(cummer(token)))))))

        if decision(add_extra_ed_probability):
            fucked_token = add_extra_ed(fucked_token)

        if decision(random_ending_y_probability):
            fucked_token = add_ending_y(fucked_token)

        # TODO: likely making fu@k into k
        # TODO: NOTE: indeed it is doing this fu@k
        #   >>>list(TextBlob("fu@k").words)
        #   ['fu', 'k']
        if add_random_plurals and decision(add_random_plurals_probability):
            fucked_token = Word(fucked_token).pluralize()

        if randomly_lemmatize and decision(randomly_lemmatize_probability):
            fucked_token = Word(fucked_token).lemmatize()

        if randomly_capitalize_word and decision(randomly_capitalize_word_probability):
            fucked_token = fucked_token.upper()

        if randomly_spongebob_word and decision(randomly_spongebob_word_probability):
            fucked_token = generate_spongebob_text(fucked_token)

        if randomly_overemphasis_punctuation and decision(
            randomly_overemphasis_punctuation_probability
        ):
            fucked_token = over_emphasise_punctuation(
                fucked_token, randomly_overemphasis_punctuation_max_fuck
            )

        if decision(common_misspellings_probability):
            fucked_token = common_mispellings(fucked_token)

        if randomly_swap_char and decision(randomly_swap_char_probability):
            fucked_token = random_swap_char(
                fucked_token, randomly_swap_char_swap_percent
            )

        if randomly_insert_char and decision(randomly_insert_char_probability):
            fucked_token = random_insert_char(
                fucked_token, randomly_insert_char_insert_percent
            )
        if decision(utf_8_char_swaps_probability):
            fucked_token = utf_8_char_swaps(fucked_token)

        if random_leet_speak and decision(random_leet_speak_probability):
            fucked_token = token_to_leet(fucked_token)

        if decision(common_misspellings_probability):
            fucked_token = common_mispellings(fucked_token)

        # TODO: likely also breaking the spacing between punctuation kittly 1!
        # TODO: `fucked` went to `DS` investigate
        # TODO: likely this is at fault
        if decision(homofiy_probability):
            fucked_token = homoify(fucked_token, homofiy_probability)

        fucked_tokens.append(fucked_token)

        if decision(add_x3_if_token_has_rawr_probability) and (
            "rawr" in fucked_token.lower()
        ):
            fucked_tokens.append("X3" if decision(0.5) else "x3")

        if decision(adding_ending_ksksk_andioop_probability) and (
            fucked_token.lower().endswith("ksk")
            or fucked_token.lower().endswith("sks")
            or "ksksk" in fucked_token.lower()
            or "sksks" in fucked_token.lower()
        ):
            for i in range(random.randint(1, 2)):
                fucked_tokens.append(recumpile_token("andioop"))
        if decision(adding_ending_ksksk_save_the_turtles_probability) and (
            fucked_token.lower().endswith("ksk")
            or fucked_token.lower().endswith("sks")
            or "ksksk" in fucked_token.lower()
            or "sksks" in fucked_token.lower()
        ):
            fucked_tokens.append(recumpile_text("save the turtles!"))

        if decision(fucking_normies_addition) and "reee" in fucked_token.lower():
            fucked_tokens.append(recumpile_text("fucking normies!"))

        if decision(get_rhymes_probability):
            for rhyme in get_runon_of_rhymes(token, max_runon=max_runon_rhymes):
                fucked_rhyme = recumpile_token(rhyme)
                fucked_tokens.append(fucked_rhyme)

        if relevant_emoji:
            fucked_tokens.append(relevant_emoji)

    for i, fucked_token in enumerate(fucked_tokens):
        if decision(space_gap_text_probability):
            # TODO: this modification may be better placed elsewhere
            fucked_token = space_gap_text(
                fucked_token,
                min_gap_size=space_gap_text_min_gap_size,
                max_gap_size=space_gap_text_max_gap_size,
            )
        # TODO: discord format options
        if decision(bold_text_probability):
            fucked_token = bold_text(fucked_token)
        elif decision(back_tick_text_probability):
            fucked_token = back_tick_text(fucked_token)
        fucked_tokens[i] = fucked_token

    return " ".join(fucked_tokens)


@logged_mutator
def bold_text(token: str) -> str:
    if not token.strip(
        string.punctuation
    ):  # don't bold tokens of all punctuation as it bugs up rejoining punctuation later *todo: maybe alternate fix?
        return token
    return f"**{token.strip('*')}**"


@logged_mutator
def get_random_rp_action_sentence() -> str:
    more_verbs = []
    more_verbs_probability = 1
    while True:
        if decision(more_verbs_probability):
            additional_verb = get_random_action_verb()
            if decision(0.5):  # TODO: config
                additional_verb = Word(additional_verb).lemmatize()
            additional_verb = recumpile_token(additional_verb)
            additional_verb = Word(additional_verb).pluralize()
            more_verbs.append(additional_verb)
        else:
            break
        more_verbs_probability -= more_verbs_probability_decay

    noun = get_random_rp_pronoun()
    if decision(0.5):  # TODO: config
        noun = Word(noun).lemmatize()

    # TODO: add boolean for enable
    noun = recumpile_token(noun)
    noun = Word(noun).pluralize()
    return to_rp_text(f"{' and '.join(more_verbs)}{' ' if more_verbs else ''}{noun}")


@logged_mutator
def lazy_char_subbing(token: str) -> str:
    """e.g.you -> u are -> r"""
    # TODO: better capital replacement

    # you -> u, yuu
    token = re.sub(
        "^y+(o+)?u+$",
        lambda match: f"u" if decision(0.5) else f"y{'u' * random.randint(1, 4)}",
        token,
        flags=re.IGNORECASE,
    )

    # are -> r, arrr
    token = re.sub(
        "^a+(r+)?e+$",
        lambda match: f"r" if decision(0.5) else f"a{'r' * random.randint(1, 4)}",
        token,
        flags=re.IGNORECASE,
    )

    # with -> wif
    token = re.sub(
        "^wi+th+$",
        lambda match: f"w{'i' * random.randint(1, 4)}{'f' * random.randint(1, 4)}",
        token,
        flags=re.IGNORECASE,
    )

    # what -> wat OR wut
    if decision(0.5):
        if decision(0.5):
            token = re.sub(
                "^wha+t$",
                lambda match: f"w{random.choice(['a', 'u']) * random.randint(1, 4)}t",
                token,
                flags=re.IGNORECASE,
            )
        else:
            token = re.sub(
                "^wha+t$",
                lambda match: f"wh{'u' * random.randint(1, 4)}t",
                token,
                flags=re.IGNORECASE,
            )

    # er -> ur
    token = re.sub(
        "(e+)r",
        lambda match: f"{'u' * (len(match.group(1)) + random.randint(0, 3))}r",
        token,
        flags=re.IGNORECASE,
        count=random.randint(0, 2),
    )

    # easy -> ez
    token = re.sub(
        "^ea+s+y+$",
        lambda match: f"e{'z' * random.randint(1, 3)}",
        token,
        flags=re.IGNORECASE,
    )

    # to,too, -> 2
    token = re.sub("to+$", lambda match: f"2", token, flags=re.IGNORECASE)
    return token


# TODO: funny -> funni spells -> spellz
@logged_mutator
def common_mispellings(token: str) -> str:
    # TODO: cleanup
    token = re.sub(
        r"([^\s])y$", lambda match: f"{match.group(1)}{'i'*random.randint(1,1)}", token
    )
    token = re.sub(
        r"([^\s])Y$", lambda match: f"{match.group(1)}{'Y'*random.randint(1,2)}", token
    )
    token = re.sub(
        r"([^\s])s$", lambda match: f"{match.group(1)}{'z'*random.randint(1,2)}", token
    )
    token = re.sub(
        r"([^\s])S$", lambda match: f"{match.group(1)}{'Z'*random.randint(1,2)}", token
    )
    token = re.sub(
        r"([^\s])z$", lambda match: f"{match.group(1)}{'s'*random.randint(1,2)}", token
    )
    token = re.sub(
        r"([^\s])Z$", lambda match: f"{match.group(1)}{'S'*random.randint(1,2)}", token
    )
    token = re.sub(
        r"([eE])([iI])", lambda match: f"{match.group(2)}{match.group(1)}", token
    )
    return token


@logged_mutator
def fix_punctuation_spacing(text: str) -> str:
    # TODO: this is a meh way to solve punct being incorrectly joined should investigate
    return re.sub(
        r"([^\s]) ([!?.,]+)", lambda match: f"{match.group(1)}{match.group(2)}", text
    )


@logged_mutator
def back_tick_text(token: str) -> str:
    if not token.strip(
        string.punctuation
    ):  # don't back_tick tokens of all punctuation as it bugs up rejoining punctuation later *todo: maybe alternate fix?
        return token
    return f"`{token.strip('`')}`"


# TODO: issues with pyenchant quick
#  patch to make this function do nothing for now
@logged_mutator
def split_compound_word(token: str) -> List[str]:
    # tokens = splitter.split(str(token))
    # if isinstance(tokens, list):
    #     return tokens
    # return [tokens]
    return [token]


@logged_mutator
def add_extra_ed(token: str) -> str:
    return re.sub(
        "([a-zA-Z]{2,})(d|ed)$",
        lambda match: f"{match.group(1)}{'ed' * random.randint(1, 2)}",
        token,
        flags=re.IGNORECASE,
    )


# TODO: grabagey code duplication
@logged_mutator
def custom_censoring(swear_word: str, censor_percent: float = 0.25) -> str:
    if len(swear_word) <= 3:  # dont censor stuff this small
        return swear_word
    censor_word_list = list("@#$%*")
    swaps = int(ceil(len(swear_word) * censor_percent))
    indexes = list(range(0, len(swear_word)))
    random.shuffle(indexes)
    indexes = indexes[:swaps]

    # avoid censoring the start or end of a string if we are not completely censoring the string
    if not (len(indexes) == len(swear_word)):
        try:
            indexes.remove(0)
        except ValueError:
            pass
        try:
            indexes.remove(len(swear_word) - 1)
        except ValueError:
            pass

    for i in indexes:
        swear_word = "".join(
            [
                swear_word[w] if w != i else random.choice(censor_word_list)
                for w in range(len(swear_word))
            ]
        )
    return swear_word


@logged_mutator
def space_gap_text(token: str, min_gap_size: int = 1, max_gap_size: int = 4) -> str:
    gap_size = random.randint(min_gap_size, max_gap_size)
    token_ends = " " * (gap_size + 1)
    token = token_ends + (" " * gap_size).join(token) + token_ends
    return token


@logged_mutator
def replace_with_random_synonym(token: str) -> str:
    # TODO: fill in with all synonyms for lulz?
    # TODO: download manual dictionary
    return token


@logged_mutator
def recumpile_line(text: str) -> str:
    new_tokens = []
    for sentence in TextBlob(text).sentences:
        new_tokens += recumpile_sentence(sentence)
    out_str = TreebankWordDetokenizer().detokenize(new_tokens)
    out_str = fix_punctuation_spacing(out_str)

    return out_str


@logged_mutator
def recumpile_text(text: str) -> str:
    # TODO: preserve spacing better / Maybe use nltk tokenizers instead of a
    #  split method
    # TODO: go sentence by sentence token by token all for sentiment analysis
    lines = []

    for line in text.split("\n"):
        lines.append(recumpile_line(line))
    return "\n".join(lines)


# TODO: fuck to ck
# TODO chicken to n
