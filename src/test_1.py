# test_load_openr1_full.py
import torch
from torch.utils.data import DataLoader
from data.load_openr1 import load_openr1_subset

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    prompts = [item["prompt"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompts": prompts,
        "labels": labels,
    }


def main():
    dataset, tokenizer = load_openr1_subset(
        tokenizer_name="google/gemma-3-270m",
        split="train",
        subset_size=100,   # smaller for speed
    )

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    print(f"Dataset size: {len(dataset)}")
    for batch in dataloader:
        print("Batch input_ids shape:", batch["input_ids"].shape)
        print("Batch attention_mask shape:", batch["attention_mask"].shape)
        print("Prompts[0] (truncated):", batch["prompts"][0][:80])
        print("Labels[0] (truncated):", batch["labels"][0][:80])
        break  # just one batch

if __name__ == "__main__":
    main()
