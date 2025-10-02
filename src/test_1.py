import torch

from data.load_openr1 import load_openr1_subset
from models.load_gemma import load_gemma

# Load model + tokenizer
model, tokenizer = load_gemma(device="mps")   # "cuda" if GPU, "mps" for Apple M1/M2

# Load dataset subset
dataset, tokenizer = load_openr1_subset(tokenizer_name="google/gemma-3-270m", subset_size=100)

# Example forward pass
sample = dataset[0]
inputs = {
    "input_ids": sample["input_ids"].unsqueeze(0).to(model.device),
    "attention_mask": sample["attention_mask"].unsqueeze(0).to(model.device)
}

with torch.no_grad():
    outputs = model(**inputs)
    print("Logits shape:", outputs.logits.shape)
