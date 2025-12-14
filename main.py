from generator import LLMEngine, ModelConfig, DecodeConfig

# model_config = ModelConfig(
#     model_id="Qwen/Qwen3-1.7B-Base",
#     use_fast_tokenizer=True,
#     device_map="auto",
# )

model_config = ModelConfig(
    model_id="gpt2-large",
    use_fast_tokenizer=True,
    device_map="auto",
)

llm_engine = LLMEngine(model_config)

decode_config = DecodeConfig(
    method="top_p",
    max_new_tokens=100,
    top_p=0.9,
    temperature=2,
    seed=42,
)

print(llm_engine.generate(
    prompt_text="once upon a time,",
    dcfg=decode_config,
))
