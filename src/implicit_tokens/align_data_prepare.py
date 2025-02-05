import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Define SYSTEM_PROMPT
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate outputs using a vLLM pipeline.")
    parser.add_argument("--model", type=str, required=True, help="vLLM model identifier.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to load from HuggingFace.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (default: test).")
    parser.add_argument("--split_start_idx", type=int, default=0, help="Dataset split idx to start (default: 0).")
    parser.add_argument("--split_end_idx", type=int, default=-1, help="Dataset split idx to end (default: -1).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output dataset.")
    parser.add_argument("--k", type=int, default=1, help="Number of outputs to generate per input.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for conforming output.")
    parser.add_argument("--similarity_threshold", type=float, default=0.9, help="Similarity threshold to detect duplicate outputs.")
    return parser.parse_args()

def validate_output(output):
    """Validate that the output contains both <think> and <answer> tags."""
    if "<think>" in output and "</think>" in output:
        return True
    return False

def extract_logic_and_answer(output):
    """Extract content between <think> and </think> tags, and <answer> tag or content after </think> tag."""
    try:
        think_content = output.split("<think>")[1].split("</think>")[0].strip()
        
        if "<answer>" in output and "</answer>" in output:
            answer_content = output.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            # Extract content starting after </think> until the end
            answer_content = output.split("</think>")[1].strip()
        
        if len(think_content) == 0 or len(answer_content) == 0:
            return None, None
        
        return think_content, answer_content
    except IndexError:
        return None, None

def main():
    args = parse_args()

    # Initialize vLLM model
    llm = LLM(args.model)
    tokenizer = llm.get_tokenizer()

    # Load dataset
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.select(range(args.split_start_idx, args.split_end_idx))

    # Process dataset
    processed_data = []
    for item in tqdm(dataset, desc="Processing items"):
        problem = item["problem"]
        solution = item["solution"]
        
        # Construct messages for chat-style input
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
        
        # Convert messages to text using a tokenizer or custom function
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        batch_outputs = []
        retries = 0

        while len(batch_outputs) < args.k and retries < args.max_retries:
            retries += 1

            # Generate outputs using vLLM
            sampling_params = SamplingParams(
                temperature=0.6, max_tokens=8192, n=args.k - len(batch_outputs)
            )
            outputs = llm.generate([text], sampling_params)

            for output in outputs[0].outputs:
                generated_text = output.text.strip()

                if validate_output(generated_text):
                    think_content, answer_content = extract_logic_and_answer(generated_text)
                    if think_content and answer_content:
                        batch_outputs.append(
                            {
                                "problem": problem,
                                "solution": solution,
                                "logic_trajectory": think_content,
                                "answer": answer_content,
                            }
                        )
                        if len(batch_outputs) >= args.k:
                            break
                    else:
                        print(f"Invalid output! Retrying...")
                else:
                    print(f"Invalid output! Retrying...")

        processed_data.extend(batch_outputs[:args.k])

    # Save processed data
    output_dataset = Dataset.from_list(processed_data)
    output_dataset.save_to_disk(args.output_file)


if __name__ == "__main__":
    main()
