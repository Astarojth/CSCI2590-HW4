import os, random, re, sqlite3, string
from collections import Counter
from functools import lru_cache
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
MODEL_NAME = "google-t5/t5-small"
PROMPT_PREFIX = "translate English to SQL: "
SQL_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]+(?:_\d+)?")
SQL_TOKEN_PATTERN = re.compile(r"'[^']*'|[A-Za-z_]+_\d+|[A-Za-z_]+|\d+|!=|<=|>=|\S")
SQL_NUMERIC_PATTERN = re.compile(r"\d+")
SQL_OPERATOR_TOKENS = {"=", "!=", "<=", ">=", "<", ">"}
SQL_PUNCT_TOKENS = {".", ",", "(", ")", ";"}
SQL_ALIAS_REF_PATTERN = re.compile(r"\b([A-Za-z_]+_\d+)\.")
SQL_ALIAS_DEF_PATTERN = re.compile(r"\b([A-Za-z_]+)\s+([A-Za-z_]+_\d+)\b")
SQL_ALIAS_REPAIR_TABLES = {
    "airport",
    "date_day",
    "days",
    "fare",
    "fare_basis",
    "flight_fare",
    "flight_stop",
}
SQL_FROM_ITEM_PATTERN = re.compile(r"^\s*([A-Za-z_]+)\s+([A-Za-z_]+_\d+)\s*$")
SQL_BARE_NUMERIC_CONDITION_PATTERN = re.compile(
    r"(?P<prefix>(?:\bWHERE\b|\bAND\b|\bOR\b|\())\s*"
    r"(?P<column>[A-Za-z_]+_\d+\.[A-Za-z_]+)\s+"
    r"(?P<value>\d+)(?=\s*(?:AND|OR|\)|$))"
)
SQL_KEYWORDS = {
    "SELECT", "DISTINCT", "FROM", "WHERE", "AND", "OR", "BETWEEN", "MIN", "MAX",
    "COUNT", "AVG", "SUM", "AS", "ON", "IN", "NOT", "EXISTS", "IS", "NULL", "LIKE",
    "ORDER", "BY", "GROUP", "HAVING", "UNION", "INTERSECT", "EXCEPT", "DESC", "ASC"
}


def normalize_nl(text):
    return " ".join(text.strip().split())


def normalize_sql(text):
    return re.sub(r"\s+", " ", text).strip()


def dedupe_from_aliases(head):
    if " FROM " not in head:
        return head

    select_part, from_part = head.split(" FROM ", 1)
    deduped_items = []
    seen_aliases = set()
    for item in from_part.split(","):
        stripped = normalize_sql(item)
        if not stripped:
            continue
        match = SQL_FROM_ITEM_PATTERN.fullmatch(stripped)
        if match:
            alias = match.group(2)
            if alias in seen_aliases:
                continue
            seen_aliases.add(alias)
        deduped_items.append(stripped)

    if not deduped_items:
        return select_part
    return f"{select_part} FROM {', '.join(deduped_items)}"


def infer_missing_operator(prefix, column):
    comparison_matches = re.findall(
        rf"{re.escape(column)}\s*(>=|>|<=|<)\s*\d+",
        prefix,
    )
    for operator in reversed(comparison_matches):
        if operator in {">", ">="}:
            return "<="
        if operator in {"<", "<="}:
            return ">="
    return "="


def fill_missing_numeric_comparisons(sql):
    def repl(match):
        prefix = match.group("prefix")
        column = match.group("column")
        value = match.group("value")
        left_context = sql[:match.start()]
        operator = infer_missing_operator(left_context, column)
        return f"{prefix} {column} {operator} {value}"

    return SQL_BARE_NUMERIC_CONDITION_PATTERN.sub(repl, sql)


def strip_truncated_sql_tail(sql):
    cleaned = sql
    previous = None
    cleanup_patterns = [
        r"\s+(?:AND|OR)\s+[A-Za-z_]+_\d+\.[A-Za-z_]+\s*(?:=|!=|<=|>=|<|>)\s*(?=\)|$)",
        r"\s+(?:AND|OR)\s+[A-Za-z_]+_\d+\.[A-Za-z_]+\s*(?=\)|$)",
        r"\s+(?:AND|OR)\s+[A-Za-z_]+_\d+\s*(?=\)|$)",
        r"\s+[A-Za-z_]+_\d+\.[A-Za-z_]+\s*(?:=|!=|<=|>=|<|>)\s*(?=\)|$)",
    ]

    while cleaned != previous:
        previous = cleaned
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        cleaned = re.sub(r"\b(?:AND|OR)\s*\)", " )", cleaned)
        cleaned = re.sub(r"\bWHERE\s*(?=\)|$)", "", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s+,", ",", cleaned)

    return cleaned


def repair_predicted_sql(text):
    sql = normalize_sql(text)
    sql = sql.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=").replace("= =", "==")
    sql = sql.replace(
        "equipment_sequence_1.aircraft_code = equipment_sequence",
        "equipment_sequence_1.aircraft_code_sequence = flight_1.aircraft_code_sequence",
    )
    sql = re.sub(r"(airport_service_\d+)\.airport_(?=\b|\s|\)|$)", r"\1.airport_code", sql)

    if " WHERE " in sql:
        head, tail = sql.split(" WHERE ", 1)
        head = dedupe_from_aliases(head)
        defined_aliases = {alias for _, alias in SQL_ALIAS_DEF_PATTERN.findall(head)}
        referenced_aliases = sorted(set(SQL_ALIAS_REF_PATTERN.findall(sql)) - defined_aliases)
        missing_aliases = []
        for alias in referenced_aliases:
            table = re.sub(r"_\d+$", "", alias)
            if table in SQL_ALIAS_REPAIR_TABLES:
                missing_aliases.append(f"{table} {alias}")
        if missing_aliases:
            head = f"{head} , {', '.join(dict.fromkeys(missing_aliases))}"
        head = dedupe_from_aliases(head)
        sql = f"{head} WHERE {tail}"

    sql = fill_missing_numeric_comparisons(sql)
    sql = strip_truncated_sql_tail(sql)

    paren_gap = sql.count("(") - sql.count(")")
    if paren_gap > 0:
        sql = f"{sql}{' )' * paren_gap}"
    while paren_gap < 0 and sql.rstrip().endswith(")"):
        sql = sql.rstrip()
        sql = sql[:-1].rstrip()
        paren_gap += 1
    return normalize_sql(sql)


def normalize_literal_key(text):
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


@lru_cache(maxsize=1)
def build_literal_candidate_map(data_folder):
    candidates = set()

    db_path = os.path.join(data_folder, "flight_database.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    for table_name in tables:
        cur.execute(f"PRAGMA table_info({table_name})")
        for _, column_name, column_type, *_ in cur.fetchall():
            column_type = (column_type or "").upper()
            if not any(tag in column_type for tag in ("CHAR", "TEXT", "CLOB")):
                continue
            try:
                cur.execute(
                    f"SELECT DISTINCT {column_name} FROM {table_name} "
                    f"WHERE {column_name} IS NOT NULL"
                )
                for (value,) in cur.fetchall():
                    if isinstance(value, str) and value.strip():
                        candidates.add(value.strip())
            except sqlite3.Error:
                continue

    conn.close()

    train_sql_path = os.path.join(data_folder, "train.sql")
    for query in load_lines(train_sql_path):
        for literal in re.findall(r"'([^']*)'", query):
            if literal.strip():
                candidates.add(literal.strip())

    mapping = {}
    for candidate in candidates:
        key = normalize_literal_key(candidate)
        if key:
            mapping.setdefault(key, set()).add(candidate)

    return mapping


def canonicalize_literal(raw_literal, literal_candidate_map):
    collapsed = re.sub(r"\s+", "", raw_literal).strip()
    if not collapsed:
        return raw_literal.strip()

    if collapsed.isdigit():
        return collapsed

    literal_key = normalize_literal_key(collapsed)
    candidates = literal_candidate_map.get(literal_key)
    if candidates and len(candidates) == 1:
        return next(iter(candidates))

    return collapsed


def is_sql_standalone_token(token):
    return (
        token in SQL_KEYWORDS
        or token in SQL_OPERATOR_TOKENS
        or token in SQL_PUNCT_TOKENS
        or bool(SQL_IDENTIFIER_PATTERN.fullmatch(token))
        or bool(SQL_NUMERIC_PATTERN.fullmatch(token))
    )


def format_sql_tokens(tokens):
    pieces = []
    for token in tokens:
        if token == ".":
            if pieces:
                pieces[-1] = pieces[-1].rstrip()
            pieces.append(".")
            continue
        if token == ",":
            if pieces:
                pieces[-1] = pieces[-1].rstrip()
            pieces.append(", ")
            continue
        if token == "(":
            pieces.append("( ")
            continue
        if token == ")":
            if pieces:
                pieces[-1] = pieces[-1].rstrip()
            pieces.append(" )")
            continue
        if token == ";":
            if pieces:
                pieces[-1] = pieces[-1].rstrip()
            pieces.append(";")
            continue
        if token in SQL_OPERATOR_TOKENS:
            if pieces and not pieces[-1].endswith(" "):
                pieces.append(" ")
            pieces.append(token)
            pieces.append(" ")
            continue
        if pieces and not pieces[-1].endswith((" ", ".", "(")):
            pieces.append(" ")
        pieces.append(token)

    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def decode_sql_sequences(tokenizer, sequences, augment_sql_vocab=False):
    if not augment_sql_vocab:
        return tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    decoded = []
    special_tokens = set(tokenizer.all_special_tokens)
    literal_candidate_map = build_literal_candidate_map(os.path.join(os.path.dirname(__file__), "data"))
    for sequence in sequences:
        tokens = tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=False)
        sql_tokens = []
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            if tok in special_tokens:
                idx += 1
                continue
            if tok == "▁":
                idx += 1
                continue

            boundary = tok.startswith("▁")
            piece = tok[1:] if boundary else tok
            if not piece:
                idx += 1
                continue

            if piece == "'":
                idx += 1
                literal_pieces = []
                while idx < len(tokens):
                    inner_tok = tokens[idx]
                    if inner_tok in special_tokens:
                        idx += 1
                        continue
                    if inner_tok == "▁":
                        idx += 1
                        continue
                    inner_piece = inner_tok[1:] if inner_tok.startswith("▁") else inner_tok
                    if inner_piece == "'":
                        break
                    literal_pieces.append(inner_piece)
                    idx += 1

                sql_tokens.append(f"'{canonicalize_literal(''.join(literal_pieces), literal_candidate_map)}'")
                if idx < len(tokens):
                    idx += 1
                continue

            if piece in SQL_PUNCT_TOKENS or piece in SQL_OPERATOR_TOKENS:
                sql_tokens.append(piece)
            elif not sql_tokens:
                sql_tokens.append(piece)
            elif not boundary and sql_tokens[-1].isdigit() and piece.isdigit():
                sql_tokens[-1] += piece
            elif boundary or is_sql_standalone_token(piece):
                sql_tokens.append(piece)
            else:
                sql_tokens[-1] += piece

            idx += 1

        decoded.append(format_sql_tokens(sql_tokens))

    return decoded

def collect_sql_identifier_tokens(data_folder):
    sql_path = os.path.join(data_folder, "train.sql")
    vocab = set()
    for sql in load_lines(sql_path):
        for tok in SQL_TOKEN_PATTERN.findall(sql):
            if SQL_IDENTIFIER_PATTERN.fullmatch(tok):
                vocab.add(tok)
    return sorted(vocab)


def build_tokenizer(data_folder, augment_sql_vocab=False):
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    if augment_sql_vocab:
        tokenizer.add_tokens(collect_sql_identifier_tokens(data_folder))
    return tokenizer


class T5Dataset(Dataset):

    def __init__(self, data_folder, split, tokenizer=None):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = tokenizer if tokenizer is not None else T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        questions = load_lines(nl_path)
        sql_queries = None if split == "test" else load_lines(os.path.join(data_folder, f"{split}.sql"))

        examples = []
        for idx, question in enumerate(questions):
            encoder_text = f"{PROMPT_PREFIX}{normalize_nl(question)}"
            encoder_tokens = tokenizer(encoder_text, truncation=True, add_special_tokens=True)
            example = {
                "encoder_ids": torch.tensor(encoder_tokens["input_ids"], dtype=torch.long),
                "encoder_mask": torch.tensor(encoder_tokens["attention_mask"], dtype=torch.long),
                "initial_decoder_input": torch.tensor([self.decoder_start_token_id], dtype=torch.long),
                "question": question,
            }

            if sql_queries is not None:
                normalized_sql = normalize_sql(sql_queries[idx])
                decoder_ids = tokenizer(normalized_sql, truncation=True, add_special_tokens=True)["input_ids"]
                if decoder_ids[-1] != tokenizer.eos_token_id:
                    decoder_ids.append(tokenizer.eos_token_id)
                example["target_ids"] = torch.tensor(decoder_ids, dtype=torch.long)
                example["sql"] = normalized_sql

            examples.append(example)

        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item["encoder_ids"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item["encoder_mask"] for item in batch], batch_first=True, padding_value=0)
    decoder_targets = pad_sequence([item["target_ids"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(
        [torch.cat([item["initial_decoder_input"], item["target_ids"][:-1]]) for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    initial_decoder_inputs = torch.cat([item["initial_decoder_input"] for item in batch]).unsqueeze(1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item["encoder_ids"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item["encoder_mask"] for item in batch], batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.cat([item["initial_decoder_input"] for item in batch]).unsqueeze(1)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, tokenizer=None):
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    dset = T5Dataset(data_folder, split, tokenizer=tokenizer)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, augment_sql_vocab=False):
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    tokenizer = build_tokenizer(data_folder, augment_sql_vocab=augment_sql_vocab)
    train_loader = get_dataloader(batch_size, "train", tokenizer=tokenizer)
    dev_loader = get_dataloader(test_batch_size, "dev", tokenizer=tokenizer)
    test_loader = get_dataloader(test_batch_size, "test", tokenizer=tokenizer)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = [f"{PROMPT_PREFIX}{normalize_nl(x)}" for x in load_lines(os.path.join(data_folder, "train.nl"))]
    train_y = [normalize_sql(x) for x in load_lines(os.path.join(data_folder, "train.sql"))]
    dev_x = [f"{PROMPT_PREFIX}{normalize_nl(x)}" for x in load_lines(os.path.join(data_folder, "dev.nl"))]
    dev_y = [normalize_sql(x) for x in load_lines(os.path.join(data_folder, "dev.sql"))]
    test_x = [f"{PROMPT_PREFIX}{normalize_nl(x)}" for x in load_lines(os.path.join(data_folder, "test.nl"))]
    return train_x, train_y, dev_x, dev_y, test_x
