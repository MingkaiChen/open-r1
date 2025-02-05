from transformers import AutoTokenizer
import datasets

def preprocess_example(example, tokenizer, system_prompt):
    """
    For each example, we build the text that the LM sees (after the system prompt
    is added by the model):
      prompt: "User: {problem}\nAssistant: <think>{logic_trajectory}</think><answer>"
      target: "{answer}</answer>"
    We also record the token length of the prompt so that when we tokenize the full text,
    we can mask out the prompt tokens (i.e. set their labels to -100) and only compute loss on the answer.
    """
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": example["problem"]},
    #     {"role": "assistant", "content": f"<think>{example['logic_trajectory']}</think>\n<answer>"},
    # ]
    # prompt_part = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=True)
    prompt_part = f"<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{example['problem']}<｜Assistant｜><think>{example['logic_trajectory']}</think><answer>"
    if tokenizer.eos_token in prompt_part:
        prompt_part = prompt_part.replace(tokenizer.eos_token, "")
    target_part = f"{example['answer']}</answer>"
    full_text = prompt_part + target_part
    # Tokenize the prompt (without adding special tokens automatically)
    prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id in prompt_ids:
        # remove the end of sentence token
        prompt_ids.remove(tokenizer.eos_token_id)
    prompt_length = len(prompt_ids)
    return {"full_text": full_text, "prompt_length": prompt_length}

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
print(tokenizer.pad_token)
exit()
dataset = datasets.load_from_disk("data/merged")
system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

output = preprocess_example(dataset[0], tokenizer, system_prompt)

# print(dataset[0]['logic_trajectory'])

think_token_id = tokenizer.convert_tokens_to_ids("<think>")
end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

ids = tokenizer(output["full_text"], return_tensors="pt")["input_ids"][0]

think_positions = (ids == think_token_id).nonzero(as_tuple=True)[0]
end_think_positions = (ids == end_think_token_id).nonzero(as_tuple=True)[0]

start_idx = think_positions[2].item()

end_idx_candidates = end_think_positions[end_think_positions > start_idx]
end_idx = end_idx_candidates[-1].item()

print(start_idx)
print(end_idx)

print(tokenizer.decode(ids[start_idx+1:end_idx]))