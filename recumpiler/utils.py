#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utilities for recumpiler"""

import csv
import os
import random
import sqlite3
from typing import List

import nltk
import pandas
from nltk.tokenize.casual import (
    _replace_html_entities,
    remove_handles,
    reduce_lengthening,
    HANG_RE,
    REGEXPS,
    EMOTICON_RE,
)
from regex import regex
from textblob.base import BaseTokenizer
from textblob.utils import strip_punc

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data")


def load_garbage_tokens() -> List[str]:
    with open(
        os.path.join(data_path, "garbage_tokens.csv"), newline="\n", encoding="utf-8"
    ) as csvfile:
        return list([item for sublist in csv.reader(csvfile) for item in sublist])


def load_simple_text_emojis() -> List[str]:
    with open(
        os.path.join(data_path, "simple_text_emoji.csv"), newline="\n", encoding="utf-8"
    ) as csvfile:
        return list([item for sublist in csv.reader(csvfile) for item in sublist])


def load_action_verbs() -> List[str]:
    with open(
        os.path.join(data_path, "action_verbs.csv"), newline="\n", encoding="utf-8"
    ) as csvfile:
        return list([item for sublist in csv.reader(csvfile) for item in sublist])


def load_rp_pronouns() -> List[str]:
    with open(
        os.path.join(data_path, "rp_pronouns.csv"), newline="\n", encoding="utf-8"
    ) as csvfile:
        return list([item for sublist in csv.reader(csvfile) for item in sublist])


def init_emoji_database() -> sqlite3.Connection:
    emoji_database = sqlite3.connect(":memory:")
    with open(
        os.path.join(
            data_path, "emoji-sentiment-data", "Emoji_Sentiment_Data_v1.0.csv"
        ),
        newline="\n",
        encoding="utf-8",
    ) as csvfile:
        df = pandas.read_csv(csvfile)
        df.to_sql(
            "Emoji_Sentiment_Data", emoji_database, if_exists="append", index=False
        )
        return emoji_database


def get_emoji_database():
    emoji_database = sqlite3.connect(":memory:")
    with open(
        os.path.join(
            data_path, "emoji-sentiment-data", "Emoji_Sentiment_Data_v1.0.csv"
        ),
        newline="\n",
        encoding="utf-8",
    ) as csvfile:
        df = pandas.read_csv(csvfile)
        df.to_sql("Emoji_Sentiment_Data", emoji_database, if_exists="fail", index=False)
    return emoji_database


def load_text_face_emoji() -> List[str]:
    with open(
        os.path.join(data_path, "emoji.csv"), newline="\n", encoding="utf-8"
    ) as csvfile:
        return list([item for sublist in csv.reader(csvfile) for item in sublist])

def split_word(word: str) -> str:
    return [char for char in word]


def decision(probability) -> bool:
    return random.random() < probability


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
