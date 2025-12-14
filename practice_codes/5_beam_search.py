import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Same structure previous codes, but uses BEAM SEARCH instead.
As Beam search is harder to implement manually, we use the built-in generate function with beam search parameters.
"""

def generate_manual(model, tok, prompt, max_new_tokens=50, num_beams=5):
    enc = tok(prompt, return_tensors="pt")
    device = model.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,   # optional for single prompt, but safe
            do_sample=False,                 # beam/greedy family
            num_beams=num_beams,             # beam width
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    return tok.decode(out_ids[0], skip_special_tokens=True)


model_id = "gpt2-large"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

prompt = "Once upon a time,"
print(generate_manual(model, tok, prompt, max_new_tokens=50, num_beams=5))
