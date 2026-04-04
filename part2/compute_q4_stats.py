import argparse
import json
import os
from statistics import mean

from transformers import T5TokenizerFast

from load_data import PROMPT_PREFIX, normalize_nl, normalize_sql, load_lines

MODEL_NAME = "google-t5/t5-small"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def tokenize_lengths(tokenizer, texts):
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


def vocab_size(tokenizer, texts):
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
    vocab = set()
    for ids in encoded["input_ids"]:
        vocab.update(ids)
    return len(vocab)


def collect_split(tokenizer, split, preprocessed):
    nl_lines = load_lines(os.path.join(DATA_DIR, f"{split}.nl"))
    sql_lines = load_lines(os.path.join(DATA_DIR, f"{split}.sql"))

    if preprocessed:
        nl_texts = [f"{PROMPT_PREFIX}{normalize_nl(text)}" for text in nl_lines]
        sql_texts = [normalize_sql(text) for text in sql_lines]
    else:
        nl_texts = nl_lines
        sql_texts = sql_lines

    return {
        "num_examples": len(nl_lines),
        "mean_sentence_length": mean(tokenize_lengths(tokenizer, nl_texts)),
        "mean_sql_length": mean(tokenize_lengths(tokenizer, sql_texts)),
        "vocab_size_nl": vocab_size(tokenizer, nl_texts),
        "vocab_size_sql": vocab_size(tokenizer, sql_texts),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Q4 statistics with the T5 tokenizer.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    stats = {
        "before": {
            "train": collect_split(tokenizer, "train", preprocessed=False),
            "dev": collect_split(tokenizer, "dev", preprocessed=False),
        },
        "after": {
            "train": collect_split(tokenizer, "train", preprocessed=True),
            "dev": collect_split(tokenizer, "dev", preprocessed=True),
        },
    }

    if args.format == "json":
        print(json.dumps(stats, indent=2))
        return

    for stage in ["before", "after"]:
        print(stage.upper())
        print("| Statistic | Train | Dev |")
        print("| --- | ---: | ---: |")
        train_stats = stats[stage]["train"]
        dev_stats = stats[stage]["dev"]
        labels = [
            ("num_examples", "Number of examples"),
            ("mean_sentence_length", "Mean sentence length"),
            ("mean_sql_length", "Mean SQL query length"),
            ("vocab_size_nl", "Vocabulary size (natural language)"),
            ("vocab_size_sql", "Vocabulary size (SQL)"),
        ]
        for key, label in labels:
            train_value = train_stats[key]
            dev_value = dev_stats[key]
            if isinstance(train_value, float):
                train_value = f"{train_value:.2f}"
                dev_value = f"{dev_value:.2f}"
            print(f"| {label} | {train_value} | {dev_value} |")
        print()


if __name__ == "__main__":
    main()
