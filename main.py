from generator import LLMEngine, ModelConfig, DecodeConfig

model_config = ModelConfig(
    model_id="Qwen/Qwen3-1.7B-Base",
    use_fast_tokenizer=True,
    device_map="auto",
)

# model_config = ModelConfig(
#     model_id="gpt2-large",
#     use_fast_tokenizer=True,
#     device_map="auto",
# )

llm_engine = LLMEngine(model_config)

decode_config = DecodeConfig(
    method="beam_search",
    max_new_tokens=100,
    t=.5,
    k=5,
    p=0.5,
    num_beams=200,
    seed=44,
)

print(llm_engine.generate(
    text_list=["once upon a time,"],
    dcfg=decode_config,
))
