#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pytests for :mod:`recumpiler.mutators`"""


import pytest

from recumpiler.__main__ import seed_random
from recumpiler.mutators import recumpile_text, owoer


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "text",
    [
        "",
        "foo bar",
        "Hey there guys today I'm making a cool python script that recompiles text into a more unreadable heap of garbage. Let me know what you guys think of it!",
        "crush my cock with a rock i must, maximum pain i must endure. okay here we go *crumpushh* oooooooooooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "hello",
        """What the fuck did you just fucking say about me, you little bitch? I'll have you know I graduated top of my class in the Navy Seals, and I've been involved in numerous secret raids on Al-Quaeda, and I have over 300 confirmed kills. I am trained in gorilla warfare and I'm the top sniper in the entire US armed forces. You are nothing to me but just another target. I will wipe you the fuck out with precision the likes of which has never been seen before on this Earth, mark my fucking words. You think you can get away with saying that shit to me over the Internet? Think again, fucker. As we speak I am contacting my secret network of spies across the USA and your IP is being traced right now so you better prepare for the storm, maggot. The storm that wipes out the pathetic little thing you call your life. You're fucking dead, kid. I can be anywhere, anytime, and I can kill you in over seven hundred ways, and that's just with my bare hands. Not only am I extensively trained in unarmed combat, but I have access to the entire arsenal of the United States Marine Corps and I will use it to its full extent to wipe your miserable ass off the face of the continent, you little shit. If only you could have known what unholy retribution your little "clever" comment was about to bring down upon you, maybe you would have held your fucking tongue. But you couldn't, you didn't, and now you're paying the price, you goddamn idiot. I will shit fury all over you and you will drown in it. You're fucking dead, kiddo.""",
        "reeeeeee reeee reeee reeee reeere erereee",
        "ksksks kkskskss kkkksksksksk kkkskskskks kkkkksksksk kksksksk sksksks ksksksks",
        "not not noot nooot not not noot not not not nothing not",
        "knot knot knot knot knot",
        "apple " * 20,
    ],
)
def test_mutate_text_blob(text):
    out_str = recumpile_text(text)
    assert isinstance(out_str, str)
    print(out_str)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "text",
    [
        """
        hello
        there
        """,
        "foo\nbar",
        "foo\n\nbar",
        "foo\n\tbar",
    ],
)
def test_mutate_text_blob_preserve_newlines(text):
    out_str = recumpile_text(text)
    assert isinstance(out_str, str)
    assert "\n" in out_str
    print(out_str)


@pytest.mark.parametrize(
    "token, expected",
    [
        ("o", "o"),
        ("ow", "owo"),
        ("owo", "owo"),
        ("owd", "owod"),
        ("owfuckme", "owofuckme"),
        ("fuckyouow", "fuckyouowo"),
        ("fuckyouowfuckme", "fuckyouowofuckme"),
        ("", ""),
    ],
)
def test_owoer(token, expected):
    seed_random("2321")
    assert owoer(token) == expected
