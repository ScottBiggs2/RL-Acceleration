# tests/test_load_openr1.py (or run interactively)
from src.data.load_openr1 import load_openr1_subset # ignore this error, it's wrong. 

ds, tok = load_openr1_subset(subset_size=10)
print("Columns:", ds.column_names)
print("First example keys:", ds[0].keys())
print("Prompt (truncated):", ds[0]["prompt"][:200])
print("Label (truncated):", ds[0]["label"][:200])
print("input_ids len:", len(ds[0]["input_ids"]))
print("attention_mask len:", len(ds[0]["attention_mask"]))
