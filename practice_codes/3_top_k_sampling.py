import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
In this code, we use the 2_minimal_loop_and_temperature.py code,
and we replace previous sampling with top-k sampling to further control randomness in token selection.
"""

def top_k_filter(logits, k: int):
    values, _ = torch.topk(logits, k)
    cutoff = values[-1]
    filtered = logits.clone()
    filtered[filtered < cutoff] = -float("inf")
    return filtered

def sample_next_token_topk(logits, top_k):
    logits = top_k_filter(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_manual(model, tok, prompt, max_new_tokens=50, top_k=0):
    enc = tok(prompt, return_tensors="pt")
    device = model.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]  # last token logits: [|V|]
            next_id = sample_next_token_topk(logits, top_k=top_k)  # [1]
            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

            if next_id.item() == tok.eos_token_id:
                break

    return tok.decode(input_ids[0], skip_special_tokens=True)


model_id = "gpt2-large"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    
prompt = "Once upon a time,"
enc = tok(prompt, return_tensors="pt")
device = model.device

print(generate_manual(model, tok, prompt, max_new_tokens=50, top_k=3))