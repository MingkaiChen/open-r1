from transformers import AutoModelForCausalLM, PreTrainedModel

model = AutoModelForCausalLM.from_pretrained("gpt2")