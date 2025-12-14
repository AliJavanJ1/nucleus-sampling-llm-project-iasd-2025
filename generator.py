from dataclasses import dataclass
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange

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
        t: float = 1.0,
        k: int = 0,
        p: float = 1.0,
        num_beams: int = 1,
        seed: Optional[int] = None,
    ):
        # methods = {"temperature", "beam_search", "top_k", "top_p", "greedy", "pure_sampling"}
        self.method: str = method
    
        self.max_new_tokens: int = max_new_tokens
    
        # parameters for sampling methods
        self.t: float = t
        self.k: int = k
        self.p: float = p
        self.num_beams: int = num_beams
        
        # reproducibility
        self.seed: Optional[int] = seed
    
    def sample_next_token(self, logits): 
        # logits: [B, V]
        if self.method == "pure_sampling":
            t = 1.0
        else:
            t = self.t
        if t is not None and t != 1.0:
            if t <= 0:
                raise ValueError("temperature (t) must be > 0 for sampling")
            logits = logits / t
        if self.method == "top_k":
            logits = self.__top_k_filter(logits)
        elif self.method == "top_p":
            logits = self.__top_p_filter(logits)
        
        probs = torch.softmax(logits, dim=-1)                     # [B, V]
        next_ids = torch.multinomial(probs, num_samples=1)        # [B, 1]
        return next_ids
    
    def __top_k_filter(self, logits):
        # logits: [B, V]
        if self.k is None or self.k <= 0:
            return logits
        values, _ = torch.topk(logits, self.k, dim=-1)   # [B, k]
        cutoff = values[..., -1].unsqueeze(-1)      # [B, 1]
        filtered = logits.clone()
        filtered[filtered < cutoff] = -float("inf")
        return filtered
    
    def __top_p_filter(self, logits):
        # logits: [B, V]
        if self.p is None or self.p >= 1.0:
            return logits
        if not (0.0 < self.p < 1.0):
            raise ValueError("top_p must be in (0,1)")

        probs = torch.softmax(logits, dim=-1)                                   # [B, V]
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)   # [B, V]
        cum = torch.cumsum(sorted_probs, dim=-1)                                # [B, V]   

        # Remove tokens where cumulative mass exceeds p, but keep the first token that crosses p
        sorted_to_remove = cum > self.p                            # [B, V]
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False

        # Scatter mask back to original vocab order
        to_remove = torch.zeros_like(sorted_to_remove).scatter(1, sorted_idx, sorted_to_remove)

        filtered = logits.clone()
        filtered[to_remove] = -float("inf")
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
        
        # for batching
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

    def generate(self, text_list: list[str], dcfg: DecodeConfig) -> list[str]:
        if dcfg.method not in {"temperature", "top_k", "top_p", "beam_search", "greedy", "pure_sampling"}:
            raise ValueError(f"Unknown method: {dcfg.method}")
        if dcfg.seed is not None:
            torch.manual_seed(dcfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(dcfg.seed)
        
        enc = self.tokenizer(text_list, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        attention_mask = attention_mask.to(device)
        
        eos_id = self.tokenizer.eos_token_id
        
        with torch.inference_mode():
            if dcfg.method in {"beam_search", "greedy"}:
                if dcfg.method == "beam_search" and dcfg.num_beams < 2:
                    raise ValueError("num_beams must be >= 2 for beam search")
                out_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,                 # beam/greedy family
                    num_beams=dcfg.num_beams if dcfg.method == "beam_search" else 1,
                    max_new_tokens=dcfg.max_new_tokens,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                    use_cache=True,
                )
            else:
                out_ids = input_ids                                 # [B, T]
                finished = torch.zeros(out_ids.size(0), dtype=torch.bool, device=device)
                
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                
                past = out.past_key_values                           # KV cache
                logits = out.logits[:, -1, :]                         # [B, V]
                
                for _ in trange(dcfg.max_new_tokens, desc="Generating", unit="tok"):
                    next_ids = dcfg.sample_next_token(logits)         # [B, 1]
                    if eos_id is not None:
                        next_ids = torch.where(
                            finished.unsqueeze(1),
                            torch.full_like(next_ids, eos_id),
                            next_ids
                        )
                    out_ids = torch.cat([out_ids, next_ids], dim=1)

                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))],
                        dim=1
                    )
                    
                    finished = finished | (next_ids.squeeze(1) == eos_id)
                    if torch.all(finished):
                        break
                    
                    out = self.model(
                        input_ids=next_ids,               # [B, 1]
                        attention_mask=attention_mask,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]         # [B, V]
            
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
