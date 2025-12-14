import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
This code uses the 1_load_model_and_generate_greedily.py code and instead of using the built-in generate function,
it manually generates text from a given prompt using a simple loop and sampling.
We use temperature sampling to introduce randomness in token selection.
"""

def sample_next_token(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # shape [1]

def generate_manual(model, tok, prompt, max_new_tokens=50, temperature=1.0):
    enc = tok(prompt, return_tensors="pt")
    device = model.device
    input_ids = enc["input_ids"].to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :]  # last token logits: [|V|]
            next_id = sample_next_token(logits, temperature=temperature)  # [1]
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

print(generate_manual(model, tok, prompt, max_new_tokens=50, temperature=1))