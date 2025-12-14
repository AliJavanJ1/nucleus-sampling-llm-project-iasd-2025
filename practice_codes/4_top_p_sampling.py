import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
In this code, we use the 3_top_k_sampling.py code,
and we replaced top-k sampling with top-p sampling.
"""

def top_p_filter(logits, p: float):
    if p >= 1.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    # keep minimal prefix with cum > p
    keep = cum < p
    # ensure at least 1 token is kept
    keep[0] = True
    # include the first token that crosses p
    if keep.sum() < keep.numel():
        keep[keep.sum()] = True

    keep_idx = sorted_idx[keep]
    filtered = torch.full_like(logits, -float("inf"))
    filtered[keep_idx] = logits[keep_idx]
    return filtered

def sample_next_token_topp(logits, top_p):
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_manual(model, tok, prompt, max_new_tokens=50, top_p=1.0):
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
            next_id = sample_next_token_topp(logits, top_p=top_p)  # [1]
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

print(generate_manual(model, tok, prompt, max_new_tokens=50, top_p=0.9))