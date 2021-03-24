#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Discord bot that recumpiles text"""

import json
import logging
from logging import getLogger, basicConfig

import discord
from discord import Message

from recumpiler.mutators import recumpile_text


basicConfig(level=logging.INFO)


__log__ = getLogger()


class RecumpilerBot(discord.Client):
    def __init__(self, *, loop=None, **options):
        discord.Client.__init__(self, loop=loop, options=options)

    async def on_ready(self):
        __log__.info(f"Logged in as username:{self.user.name} id:{self.user.id}")
        __log__.info(f"Watching channels: {channels}")

    async def on_message(self, message):
        message: Message = message
        __log__.info(
            f"obtained {message.author.id} {message.author.display_name} message: {message.content} message_type:{message.channel.type}"
        )
        if message.author.id != self.user.id:
            if message.channel.id in channels:
                await message.delete()
                # TODO: add some way to recover if `recumpile_text` fails?
                fucked_text = recumpile_text(message.content)
                # TODO: It is not impossible that recumpile_text generates text longer than 2000 characters!
                #       maybe recumpile_text should have options to limit its generated output length to avoid
                #       either dropping the message or embedding it as a file.
                __log__.info(f"fucked message text: {fucked_text}")
                await message.channel.send(
                    content=f"<@!{message.author.id}> {fucked_text}"
                )


client = RecumpilerBot()

# file containing a simple json list of the channels you want all text to be recumpiled [<channel_id>]
with open("../channels.json") as f:
    channels = json.load(f)

# file containing the discord bot token
with open("../tokens/token.txt") as f:
    token = f.read().strip()
client.run(token, bot=True)
