#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main argparse entrypoint"""

import argparse
import logging
import os
import sys
import random
from logging.handlers import TimedRotatingFileHandler

import numpy

from recumpiler.mutators import recumpile_text

LOG_LEVEL_STRINGS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def log_level(log_level_string: str):
    """argparse type function for determining the specified logging level"""
    if log_level_string not in LOG_LEVEL_STRINGS:
        raise argparse.ArgumentTypeError(
            "invalid choice: {} (choose from {})".format(
                log_level_string, LOG_LEVEL_STRINGS
            )
        )
    return getattr(logging, log_level_string, logging.INFO)


def add_log_parser(parser):
    """Add logging options to the argument parser"""
    group = parser.add_argument_group(title="Logging")
    group.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        type=log_level,
        help="Set the logging output level",
    )
    group.add_argument(
        "--log-dir",
        dest="log_dir",
        help="Enable TimeRotatingLogging at the directory " "specified",
    )
    group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )


def init_logging(args, log_file_path):
    """Intake a argparse.parse_args() object and setup python logging"""
    handlers_ = []
    log_format = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s")
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(args.log_dir, log_file_path),
            when="d",
            interval=1,
            backupCount=7,
            encoding="UTF-8",
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(args.log_level)
        handlers_.append(file_handler)
    if args.verbose:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(args.log_level)
        handlers_.append(stream_handler)

    logging.basicConfig(handlers=handlers_, level=args.log_level)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompile text to be semi-readable memey garbage."
    )

    parser.add_argument("-s", "--seed", type=str, help="Manually input a random seed.")

    group = parser.add_argument_group(title="Input")
    group = group.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="Text to recumpile.")
    group.add_argument(
        "--repl", help="Accept stdin and create recumpiler REPL.", action="store_true"
    )

    add_log_parser(parser)

    return parser


def recumpile_repl():
    while True:
        text = input("recumpiler>")
        fucked_text = recumpile_text(text)
        print(fucked_text)


def str_to_32_bit_unsigned_int(string: str) -> int:
    """convert a :obj:`str` to a 32 bit unsigned int for
    :meth:`numpy.random.seed` input"""
    string_int = ord(string[0]) << 7 if string else 0
    for c in map(ord, string):
        string_int = ((1000003 * string_int) ^ c) & 0xFFFFFFFF
    string_int ^= len(string)
    string_int = -2 if string_int == -1 else string_int
    return string_int


def str_seed_numpy_random(seed: str):
    """seed :mod:`numpy.random` with a string based seed

    numpy random takes a 32 bit unsigned int as such we use a similar method to
    how :mod:`random` to convert a string seed to a 32 bit unsigned int"""
    numpy.random.seed(str_to_32_bit_unsigned_int(seed))


def seed_random(seed: str):
    """seed both python built-in :mod:`random` and :mod:`numpy.random`"""
    str_seed_numpy_random(seed)
    random.seed(seed)


def main(argv=None):
    """main entry point"""
    args = get_parser().parse_args(argv)
    init_logging(args, "recumpiler.log")

    if args.seed is not None:
        seed_random(args.seed)

    if args.text:
        fucked_text = recumpile_text(args.text)
        print(fucked_text)
    if args.repl:
        recumpile_repl()


if __name__ == "__main__":
    main()
