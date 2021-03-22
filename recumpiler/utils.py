#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""utilities for recumpiler"""

import csv
import os
import random
import sqlite3
from typing import List

import pandas


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


def decision(probability) -> bool:
    return random.random() < probability
