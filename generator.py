from dataclasses import dataclass
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

@dataclass
class ModelConfig:
    # model_ids = {"gpt2-large", "Qwen/Qwen3-1.7B-Base"}
    model_id: str
    
    use_fast_tokenizer: bool = True
    device_map: str = "auto"


class DecodeConfig:
    def __init__(
        self,
        method: str = "top_p",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_beams: int = 1,
        seed: Optional[int] = None,
    ):
        # methods = {"temperature", "beam_search", "top_k", "top_p"}
        self.method: str = method
    
        self.max_new_tokens: int = max_new_tokens
    
        # parameters for sampling methods
        self.temperature: float = temperature
        self.top_k: int = top_k
        self.top_p: float = top_p
        self.num_beams: int = num_beams
        
        # reproducibility
        self.seed: Optional[int] = seed
    
    def sample_next_token(self, logits):
        if self.temperature is not None and self.temperature != 1.0:
            if self.temperature <= 0:
                raise ValueError("temperature must be > 0 for sampling")
            logits = logits / self.temperature
        if self.method == "top_k":
            logits = self.__top_k_filter(logits, self.top_k)
        elif self.method == "top_p":
            logits = self.__top_p_filter(logits, self.top_p)
        
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def __top_k_filter(self, logits, k: int):
        if k is None or k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        cutoff = values[-1]
        filtered = logits.clone()
        filtered[filtered < cutoff] = -float("inf")
        return filtered
    
    def __top_p_filter(self, logits, p: float):
        if p is None or p >= 1.0:
            return logits
        if not (0.0 < p < 1.0):
            raise ValueError("top_p must be in (0,1)")

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
    
class LLMEngine:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model(self.tokenizer)
        
    def _get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id, 
            use_fast=self.model_config.use_fast_tokenizer
        )
        
        # Ensure pad_token exists for generation (GPT-2 has eos but no pad)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Helpful for decoder-only models when batching
        tokenizer.padding_side = "left"
        return tokenizer
                
    def _get_model(self, tokenizer: AutoTokenizer):            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            device_map=self.model_config.device_map,
            dtype="auto",
        )
        # Propagate pad/eos into generation config if needed
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.generation_config, "eos_token_id", None) is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            
        model.eval()
        return model

    def generate(self, prompt_text: str, dcfg: DecodeConfig) -> str:    
        if dcfg.seed is not None:
            torch.manual_seed(dcfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(dcfg.seed)
        
        enc = self.tokenizer(prompt_text, return_tensors="pt")
        device = self.model.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        
        with torch.inference_mode():
            if dcfg.method == "beam_search":
                out_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,                 # beam/greedy family
                    num_beams=dcfg.num_beams,             # beam width
                    max_new_tokens=dcfg.max_new_tokens,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                for _ in range(dcfg.max_new_tokens):
                    out = self.model(input_ids=input_ids)
                    logits = out.logits[0, -1, :]  # last token logits: [|V|]
                    next_id = dcfg.sample_next_token(logits)  # [1]
                    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
                    attention_mask = None if attention_mask is None else torch.cat(
                        [attention_mask, torch.ones((1,1), dtype=attention_mask.dtype, device=device)],
                        dim=1
                    )       

                    if next_id.item() == self.tokenizer.eos_token_id:
                        break
                out_ids = input_ids
            
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                


