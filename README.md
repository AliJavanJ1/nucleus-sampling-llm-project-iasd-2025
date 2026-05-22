# nucleus-sampling-llm-project-iasd-2025

This project is a smaller replication of the experiment described in the paper [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751). Its goal is to revalidate some of the paper's findings by comparing different decoding strategies for large language model text generation, with a particular focus on nucleus (top-p) sampling.

The repository contains a small generation engine built on top of Hugging Face Transformers, plus notebooks and result files used to experiment with different models and decoding settings. The code is designed to make it easy to switch between models, tune decoding parameters, and evaluate the effect of each strategy on generated text.

## What the project does

- Loads causal language models with Hugging Face Transformers
- Supports multiple decoding methods:
  - greedy decoding
  - beam search
  - top-k sampling
  - top-p / nucleus sampling
  - temperature-based sampling
  - pure sampling
- Uses KV caching for more efficient token-by-token generation
- Produces and stores generation results for different model/decoding combinations
- Includes notebooks for experimentation and plotting

## Main files

- `generator.py`: core generation engine and decoding configuration
- `main.py`: simple example script showing how to run generation
- `main.ipynb`: notebook version of the main experiment workflow
- `main_qwen.ipynb`: experiments using a Qwen model
- `plots.ipynb`: analysis and visualization of results
- `results_*.csv`: saved outputs from generation experiments

## Models used

The repository currently includes experiments with:

- `Qwen/Qwen3-0.6B-Base`
- `gpt2-large`

## Installation

The project uses `uv` and targets Python 3.12 or later. Install dependencies with:

```bash
uv sync
```

## Example usage

`main.py` shows how to instantiate the engine and generate text:

```python
from generator import LLMEngine, ModelConfig, DecodeConfig

model_config = ModelConfig(
    model_id="Qwen/Qwen3-0.6B-Base",
    use_fast_tokenizer=True,
    device_map="auto",
)

llm_engine = LLMEngine(model_config)

decode_config = DecodeConfig(
    method="beam_search",
    max_new_tokens=100,
    num_beams=200,
    seed=44,
)

print(llm_engine.generate(
    text_list=["once upon a time,"],
    dcfg=decode_config,
))
```

## Notes

- For sampling methods, the generator applies temperature scaling and then filters logits using top-k or top-p when selected.
- Beam search and greedy decoding use the built-in `model.generate()` API.
- The project is intended as a compact reproduction of the paper's decoding comparison setup, not a full-scale benchmark.

## Project structure

```text
.
├── generator.py
├── main.py
├── main.ipynb
├── main_qwen.ipynb
├── plots.ipynb
├── results_*.csv
├── practice_codes/
└── pyproject.toml
```
