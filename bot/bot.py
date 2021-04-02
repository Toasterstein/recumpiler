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


import sys

sys.setrecursionlimit(5500)


class RecumpilerBot(discord.Client):
    def __init__(self, *, loop=None, **options):
        discord.Client.__init__(self, loop=loop, options=options)

    async def on_ready(self):
        __log__.info(f"Logged in as username:{self.user.name} id:{self.user.id}")
        __log__.info(f"Watching channels: {channels}")

    async def my_background_task(self, message, waiting_message: discord.Message):
        await self.wait_until_ready()
        try:
            original_fucked_text = recumpile_text(message.content)
            fucked_text = (
                original_fucked_text.encode("utf-8")[
                    : 2000 - (3 + len(message.author.display_name))
                ]
            ).decode("utf-8")
            # TODO: It is not impossible that recumpile_text generates text longer than 2000 characters!
            #       maybe recumpile_text should have options to limit its generated output length to avoid
            #       either dropping the message or embedding it as a file.
            __log__.info(f"fucked message text: {fucked_text}")
            await waiting_message.edit(content=f"<@!{message.author.id}> {fucked_text}")
            for attachment in message.attachments:
                # TODO: quick hack to get images reposted
                await waiting_message.channel.send(content=attachment.url)
            if original_fucked_text != fucked_text:
                raise ValueError(
                    "post-processed discord-ready fucked text not the same likely output was too long"
                )
        except:
            fucked_text = recumpile_text(
                "Oops i had a fucky wucky recumpiling your text! Your text could be too big UWU!"
            )
            await waiting_message.channel.send(
                content=f"<@!{message.author.id}> {fucked_text}"
            )

    async def on_message(self, message):
        message: Message = message
        __log__.debug(
            f"obtained {message.author.id} {message.author.display_name} message: {message.content} message_type:{message.channel.type}"
        )
        if message.author.id != self.user.id:
            if message.author.id != self.user.id:
                if isinstance(message.channel, discord.DMChannel):
                    waiting_message = await message.channel.send(
                        content=f"<@!{message.author.id}> {recumpile_text('recumpiling your text :3')}"
                    )
                    self.loop.create_task(
                        self.my_background_task(message, waiting_message)
                    )
            if message.channel.id in channels:
                if message.author.id in mods:
                    if message.content.startswith("NO:"):
                        __log__.info(f"not recumpiling for mod: {message.author.id}")
                        return
                await message.delete()
                waiting_message = await message.channel.send(
                    content=f"<@!{message.author.id}> {recumpile_text('recumpiling your text :3')}"
                )
                self.loop.create_task(self.my_background_task(message, waiting_message))


client = RecumpilerBot()

# file containing a simple json list of the channels you want all text to be recumpiled [<channel_id>]
with open("../channels.json") as f:
    channels = json.load(f)

with open("../mods.json") as f:
    mods = json.load(f)

# file containing the discord bot token
with open("../tokens/token.txt") as f:
    token = f.read().strip()
client.run(token, bot=True)
