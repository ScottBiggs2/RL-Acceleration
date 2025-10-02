import torch
from datasets import load_dataset
from src.models.load_gemma import load_gemma
from src.reward.reward_functions import math_reward
from src.mask.activation_hooks import ActivationCollector
from src.mask.cav import compute_cav, rank_neurons

def main():
    # 1. Load model and dataset
    model, tokenizer = load_gemma(device="mps")  # M1 Mac
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train[:10%]")

    # 2. Collect activations + rewards
    collector = ActivationCollector()
    layers_to_hook = [(f"block_{i}", model.model.layers[i].mlp) for i in range(3)]  # first 3 blocks
    hooks = collector.register_hooks(model, layers_to_hook)

    pos_acts, neg_acts = [], []
    for ex in ds.select(range(200)):  # start small
        input_ids = tokenizer(ex["problem"], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=128)
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        reward = math_reward(pred, ex["solution"])
        acts = collector.activations[layers_to_hook[0][0]]  # grab first hooked layer

        if reward == 1:
            pos_acts.append(acts)
        else:
            neg_acts.append(acts)

    hooks = [h.remove() for h in hooks]

    pos_acts = torch.cat(pos_acts)
    neg_acts = torch.cat(neg_acts)

    # 3. Compute CAV + rank
    cav = compute_cav(pos_acts, neg_acts)
    ranked, scores = rank_neurons(pos_acts, cav)

    print("Top 10 important neurons:", ranked[:10])

if __name__ == "__main__":
    main()
