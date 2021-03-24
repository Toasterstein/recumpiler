import json
import os

import numpy as np
import spacy

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, "data")


def load_emojis():
    rows = []
    with open(os.path.join(data_path, "emojinet.json")) as f:
        for emoji in json.loads(f.read()):
            rows.append([emoji["name"], emoji["unicode"], " ".join(emoji["keywords"])])
    return np.array(rows)


emojis = load_emojis()


# python3 -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

from tqdm import tqdm


glove_vector_file = os.path.join(data_path, "glove_vector.vec")
try:
    nlp.vocab.from_disk(glove_vector_file)
except Exception as e:
    print(f"{e} generating {glove_vector_file}")
    with open(os.path.join(data_path, "glove.6B.300d.txt"), "r", encoding="utf-8") as f:
        for line in tqdm(f, total=400000):
            parts = line.split()
            word = parts[0]
            vec = np.array([float(v) for v in parts[1:]], dtype="f")
            nlp.vocab.set_vector(word, vec)
        nlp.vocab.to_disk(glove_vector_file)


doc_vectors_file = os.path.join(data_path, "doc_vectors.npy")
try:
    doc_vectors = np.load(doc_vectors_file, allow_pickle=True)
    rebuilt = False
except Exception as e:
    print(f"{e} generating {doc_vectors_file}")
    docs = [nlp(str(keywords)) for _, _, keywords in tqdm(emojis)]
    doc_vectors = np.array([doc.vector for doc in docs])
    rebuilt = True

if rebuilt:
    np.save(doc_vectors_file, doc_vectors)


from numpy import dot
from numpy.linalg import norm


def most_similar(vectors, vec):
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))

    return np.argsort(-dst), []


def min_dist_similar(vectors, vec, min_dist: float = -0.02):
    # cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))
    distances = list(dst)
    ids = np.argwhere(-dst < min_dist).flatten()
    distances = [-distances[i] for i in ids]
    return ids, distances


def occurrence_query(v, most_n: int = 5, min_dist: float = None):
    # NOTE: -0.024 is relatively close
    ids, dsts = most_similar(doc_vectors, v)
    if most_n is not None:
        ids = ids[:most_n]

    ids_ordered_5 = ids

    # unordered dists past threshold
    if min_dist is not None:
        ids, dsts = min_dist_similar(doc_vectors, v, min_dist=min_dist)
        ids = ids.flatten()
        ids = np.array([i for i in ids_ordered_5 if i in ids])

    emoji_list = []
    if list(ids):
        # TODO: refactor out to index to emoji
        for name, unicode, keywords in emojis[ids]:
            values = unicode.split(" ")
            for v in values:
                c = chr(int(v.replace("U+", ""), 16))
                print(c, name)
                emoji_list.append(c)
    #         display(HTML('<font size="+3">{}</font>'.format(' '.join([x for x in emoji_list]))))
    else:
        print(f"no emojis found for min_dist:{min_dist}")
    return emoji_list


from textblob import TextBlob, Word


def noun_occurrence_query(noun: str, most_n: int = 5, min_dist: float = None):
    tokens = TextBlob(noun).tokens
    noun_vector = sum([nlp(str(token)).vector for token in tokens])
    return occurrence_query(noun_vector, most_n, min_dist=min_dist)


import enum


class emoji_glove_lookup_accuracy(enum.Enum):
    low = -0.001
    med = -0.0015
    high = -0.002
    very_high = -0.02
    extremely_high = -0.025


def word_emoji_inserter(word: Word):
    # check if noun type
    out_word = str(word)

    # word_emoji = emoji_glove_lookup.noun_occurrence_query(str(word))

    # TODO: seeing if word lemmization helps in lookup occurace search
    verb_lemmatized_word = Word(word).singularize().lemmatize("v").lower()
    print(f"word:{word} verb_lemmatized_word:{verb_lemmatized_word}")
    word_emoji = noun_occurrence_query(
        verb_lemmatized_word, min_dist=emoji_glove_lookup_accuracy.very_high.value
    )

    out_word = f"{word}{' ' + word_emoji[0] if word_emoji else ''}"

    return out_word


print(word_emoji_inserter("car"))


def get_word_emoji(word: Word):
    verb_lemmatized_word = Word(word).singularize().lemmatize().lower()
    word_emoji = noun_occurrence_query(
        verb_lemmatized_word, min_dist=emoji_glove_lookup_accuracy.high.value
    )
    if word_emoji:
        return word_emoji[0]
    else:
        return None


print(get_word_emoji("car"))
print(get_word_emoji("crab"))
print(get_word_emoji("gun"))
print(get_word_emoji("ship"))
print(get_word_emoji("train"))
