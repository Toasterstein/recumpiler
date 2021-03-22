#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pytests for :mod:`recumpiler.__main__`"""

import argparse

from recumpiler.__main__ import main, get_parser

import pytest


def test_get_parser():
    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "text",
    [
        "",
        "foo bar",
        "Hey there guys today I'm making a cool python script that recompiles text into a more unreadable heap of garbage. Let me know what you guys think of it!",
        "crush my cock with a rock i must, maximum pain i must endure. okay here we go *crumpushh* oooooooooooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "reeeeeee reeee reeee reeee reeere erereee",
        "ksksks kkskskss kkkksksksksk kkkskskskks kkkkksksksk kksksksk sksksks ksksksks",
        "not not noot nooot not not noot not not not nothing not",
        "knot knot knot knot knot",
        "apple " * 20,
    ],
)
@pytest.mark.parametrize("seed", ["", "foo bar", "my really long seed" * 1000,])
def test_random_seeding(text, seed, capsys):
    first_invoke_output = None
    for i in range(10):
        main(["--seed", seed, "--text", text])
        captured = capsys.readouterr()
        invoke_output = captured.out
        if first_invoke_output is None:
            first_invoke_output = invoke_output
        assert invoke_output == first_invoke_output
        with capsys.disabled():
            print(f"invocation: {i} output:", invoke_output)
