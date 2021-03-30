import emoji as emoji_lookup


def get_cheap_emoji_alias(token: str):
    emoji = emoji_lookup.emojize(
        f"Åéãíç{token}Åéãíç", use_aliases=True, delimiters=("Åéãíç", "Åéãíç")
    )
    if emoji == f"Åéãíç{token}Åéãíç" or emoji == f":{token}:":
        # we failed to find a emoji
        return None
    else:
        return emoji
