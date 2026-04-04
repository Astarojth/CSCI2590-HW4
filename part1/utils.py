import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import hashlib
import re
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "erfcxs",
    "e": "rdsw",
    "f": "rtgvcd",
    "g": "tyhbvf",
    "h": "yujnbg",
    "i": "uojk",
    "j": "uikmnh",
    "k": "iolmj",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "o": "pikl",
    "p": "ol",
    "q": "wa",
    "r": "tfde",
    "s": "wedxza",
    "t": "ygfr",
    "u": "yihj",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "uhgt",
    "z": "asx",
}
TOKEN_PATTERN = re.compile(r"^[A-Za-z]+$")
DETOKENIZER = TreebankWordDetokenizer()


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    tokens = word_tokenize(text)
    transformed_tokens = []

    for token in tokens:
        # Keep the perturbation simple and readable, but increase coverage enough
        # to produce a meaningful OOD gap on the transformed test split.
        if not TOKEN_PATTERN.match(token) or len(token) < 4 or rng.random() >= 0.35:
            transformed_tokens.append(token)
            continue

        chars = list(token)
        valid_positions = [idx for idx, ch in enumerate(chars) if ch.lower() in KEYBOARD_NEIGHBORS]
        can_swap = len(chars) > 4

        if valid_positions and (not can_swap or rng.random() < 0.7):
            pos = rng.choice(valid_positions)
            original = chars[pos]
            replacement = rng.choice(KEYBOARD_NEIGHBORS[original.lower()])
            chars[pos] = replacement.upper() if original.isupper() else replacement
        elif can_swap:
            pos = rng.randrange(len(chars) - 1)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        transformed_tokens.append("".join(chars))

    example["text"] = DETOKENIZER.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
