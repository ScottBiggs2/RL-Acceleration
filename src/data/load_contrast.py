# src/data/load_contrast.py
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import itertools

def load_contrast_subset(dataset_name, tokenizer_name="google/gemma-3-270m", subset_size=10, max_length=512):
    """
    Load a small subset of a contrast dataset for debugging.
    Falls back to a small uncompressed dataset if loading fails.

    Returns:
        examples: list of dicts with keys 'text', 'input_ids', 'attention_mask'
        tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    examples = []

    try:
        print(f"[LOAD] Streaming contrast dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train", streaming=True)
        stream = ds.iter(batch_size=1)  # <-- batch_size required for IterableDataset.iter()
        raw_examples = list(itertools.islice(stream, subset_size))
        for r in raw_examples:
            txt = r.get("text") or r.get("content") or r.get("prompt") or str(r)
            examples.append(txt)
        print(f"[LOAD] Loaded {len(examples)} examples from {dataset_name}")
    except Exception as e:
        print(f"[WARN] Could not load {dataset_name}: {e}")
        print("[LOAD] Falling back to wikitext-2-raw-v1 for debugging subset")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{subset_size}]")
        examples = [ex["text"] for ex in ds]

    # Tokenize
    toks = tokenizer(examples, truncation=True, padding="max_length", max_length=max_length)
    tokenized_dataset = [
        {"text": examples[i], "input_ids": toks["input_ids"][i], "attention_mask": toks["attention_mask"][i]}
        for i in range(len(examples))
    ]
    return tokenized_dataset, tokenizer
