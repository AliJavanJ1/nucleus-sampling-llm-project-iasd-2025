import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
In this code, we load a pre-trained language model and its tokenizer,
then generate text from a given prompt using greedy decoding.
"""

model_id = "gpt2-large"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
    
prompt = "Once upon a time,"
enc = tok(prompt, return_tensors="pt")
device = model.device
enc = {k: v.to(device) for k, v in enc.items()}

with torch.inference_mode():
    out = model.generate(
        **enc,
        do_sample=False,      # greedy/beam family
        max_new_tokens=50,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))