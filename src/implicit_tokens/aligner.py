import argparse
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import wandb

###############################################
# Define the ImplicitTokensAlignment classes  #
###############################################

class ImplicitTokensAligner(nn.Module):
    """
    This module searches for a <think>...</think> region in each example’s
    token embeddings, and replaces the tokens inside with k aggregated tokens
    via cross-attention.
    """
    def __init__(self, hidden_size: int, k: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        # k learnable implicit tokens
        self.implicit_tokens = nn.Parameter(torch.randn(k, hidden_size))
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, embeddings: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, pad_token_emb) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = embeddings.shape
        new_embeddings_list = []
        new_attention_mask_list = []

        # Get the token ids for the <think> and </think> tokens.
        think_token_id = tokenizer.convert_tokens_to_ids("<think>")
        end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

        # Process each example in the batch
        for i in range(batch_size):
            ids = input_ids[i]         # (seq_len,)
            emb = embeddings[i]        # (seq_len, hidden_size)
            mask = attention_mask[i]   # (seq_len,)

            # Find positions of <think> and </think>
            think_positions = (ids == think_token_id).nonzero(as_tuple=True)[0]
            end_think_positions = (ids == end_think_token_id).nonzero(as_tuple=True)[0]

            if len(think_positions) == 0 or len(end_think_positions) == 0:
                # If there is no <think> region, leave this example unchanged.
                new_embeddings_list.append(emb)
                new_attention_mask_list.append(mask)
                continue

            # Process only the first <think>...</think> region.
            start_idx = think_positions[2].item()
            # Find the last </think>
            end_idx_candidates = end_think_positions[end_think_positions > start_idx]
            if len(end_idx_candidates) == 0:
                new_embeddings_list.append(emb)
                new_attention_mask_list.append(mask)
                continue
            end_idx = end_idx_candidates[-1].item()

            # Get the embeddings for the region inside the tags.
            region_emb = emb[start_idx + 1 : end_idx]  # shape: (region_length, hidden_size)
            region_len = region_emb.shape[0]

            if region_len == 0:
                # If the region is empty, simply remove the tags.
                new_emb = torch.cat([emb[:start_idx], emb[end_idx + 1:]], dim=0)
                new_mask = torch.cat([mask[:start_idx], mask[end_idx + 1:]], dim=0)
                new_embeddings_list.append(new_emb)
                new_attention_mask_list.append(new_mask)
                continue

            # Prepare the implicit tokens as queries and perform cross-attention.
            queries = self.implicit_tokens.unsqueeze(1)  # (k, 1, hidden_size)
            region_emb_seq = region_emb.unsqueeze(1)       # (region_len, 1, hidden_size)
            attn_output, _ = self.cross_attention(query=queries, key=region_emb_seq, value=region_emb_seq)
            aligned_region = attn_output.squeeze(1)        # (k, hidden_size)

            # Replace the original region with the aggregated tokens.
            new_emb = torch.cat([emb[:start_idx], aligned_region, emb[end_idx + 1:]], dim=0)
            new_mask = torch.cat([
                mask[:start_idx],
                torch.ones(self.k, dtype=mask.dtype, device=mask.device),
                mask[end_idx + 1:]
            ], dim=0)
            new_embeddings_list.append(new_emb)
            new_attention_mask_list.append(new_mask)

        # Pad all sequences to the same (max) length.
        new_seq_lengths = [emb.size(0) for emb in new_embeddings_list]
        max_len = max(new_seq_lengths)
        padded_embeddings = []
        padded_masks = []
        for emb, mask in zip(new_embeddings_list, new_attention_mask_list):
            pad_len = max_len - emb.size(0)
            if pad_len > 0:
                # pad_emb = torch.zeros(pad_len, hidden_size, device=emb.device)
                pad_emb = torch.cat([pad_token_emb.unsqueeze(0)] * pad_len, dim=0)
                pad_mask = torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)
                emb = torch.cat([emb, pad_emb], dim=0)
                mask = torch.cat([mask, pad_mask], dim=0)
            padded_embeddings.append(emb.unsqueeze(0))
            padded_masks.append(mask.unsqueeze(0))
        new_embeddings = torch.cat(padded_embeddings, dim=0)
        new_attention_mask = torch.cat(padded_masks, dim=0)
        return new_embeddings, new_attention_mask


class ImplicitTokensAlignment(nn.Module):
    """
    This model wraps a pretrained causal LM (e.g. GPT‑2) and an aligner module.
    When initializing:
      - It loads the pretrained LM and its tokenizer.
      - It adds special tokens (including <think>, </think>, <answer>, </answer>, and k
        new implicit tokens).
      - It initializes the aligner module.
      - It stores an optional system prompt.
    
    In the forward pass, the model can accept either raw text (a string or list of strings)
    or a dictionary with pre-tokenized tensors. In all cases the model:
      1. (If needed) prepends the system prompt.
      2. Tokenizes the text.
      3. Computes the input embeddings.
      4. Passes them (along with token IDs) through the aligner.
      5. Feeds the resulting embeddings to the LM, optionally computing a loss if labels
         are provided.
    """
    def __init__(self, pretrained_model_name: str, k: int,
                 device: torch.device = None):
        super().__init__()

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
        
        self.config = self.model.config

        # Add special tokens required by the alignment mechanism.
        additional_special_tokens = []
        for token in ["<think>", "</think>", "<answer>", "</answer>"]:
            if token not in self.tokenizer.get_vocab():
                additional_special_tokens.append(token)
        # # Add k new implicit tokens (e.g. <implicit_0>, …, <implicit_{k-1}>)
        # implicit_tokens = [f"<implicit_{i}>" for i in range(k)]
        # for token in implicit_tokens:
        #     if token not in self.tokenizer.get_vocab():
        #         additional_special_tokens.append(token)
        # if additional_special_tokens:
        #     special_tokens_dict = {"additional_special_tokens": additional_special_tokens}
        #     self.tokenizer.add_special_tokens(special_tokens_dict)
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        
        # # Save the indices for the implicit tokens (if needed).
        # self.implicit_token_ids = self.tokenizer.convert_tokens_to_ids(implicit_tokens)
        self.hidden_size = self.model.config.hidden_size

        # Initialize the aligner.
        self.aligner = ImplicitTokensAligner(hidden_size=self.hidden_size, k=k)

        # Move to device.
        if device is not None:
            self.device = device
            self.model.to(device)
            self.aligner.to(device)
        else:
            self.device = next(self.model.parameters()).device

        # Ensure that a padding token is set (e.g. for GPT‑2, use eos_token).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def forward(self, input_ids, attention_mask, labels):
        """
        Accepts either:
          - A dictionary with keys "input_ids", "attention_mask", and optionally "labels" (for pre-tokenized batches),
          - Or a string (or list of strings) in which case tokenization is applied (and the system prompt is prepended).
        """
        # if isinstance(batch_or_text, dict) and "input_ids" in batch_or_text:
        #     input_ids = batch_or_text["input_ids"].to(self.device)
        #     attention_mask = batch_or_text["attention_mask"].to(self.device)
        #     if labels is None and "labels" in batch_or_text:
        #         labels = batch_or_text["labels"].to(self.device)
        # else:
        #     # Assume raw text input.
        #     inputs = self.tokenizer(batch_or_text, return_tensors="pt", padding=True, truncation=True)
        #     input_ids = inputs["input_ids"].to(self.device)
        #     attention_mask = inputs["attention_mask"].to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        # Compute input embeddings.
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        pad_token_emb = self.model.get_input_embeddings()(torch.tensor(self.tokenizer.pad_token_id, device=self.device))
        # Pass embeddings through the aligner.
        aligned_embeddings, new_attention_mask = self.aligner(input_embeddings, input_ids, attention_mask, self.tokenizer, pad_token_emb)
        # Forward through the LM (if labels are provided, the LM returns the loss).
        outputs = self.model(inputs_embeds=aligned_embeddings, attention_mask=new_attention_mask, labels=labels)
        return outputs

#################################################
# Data preprocessing and custom collate function  #
#################################################

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

def data_collator(batch, tokenizer):
    """
    For a list of preprocessed examples (each with keys "full_text" and "prompt_length"),
    tokenize the full_text and create labels so that tokens corresponding to the prompt are masked out.
    """
    input_ids = []
    labels = []
    for example in batch:
        tokenized = tokenizer(example["full_text"], add_special_tokens=False)
        ids = tokenized["input_ids"]
        p_len = example["prompt_length"]
        # For tokens before the answer begins (i.e. the prompt part) set label to -100
        example_labels = [-100] * p_len + ids[p_len:]
        input_ids.append(ids)
        labels.append(example_labels)
    # Pad the sequences to the same length.
    batch_inputs = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")["input_ids"]
    batch_labels = tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
    attention_mask = (batch_inputs != tokenizer.pad_token_id).long()
    return {"input_ids": batch_inputs, "attention_mask": attention_mask, "labels": batch_labels}

##########################
# Training Script (main) #
##########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the local dataset (to be loaded with datasets.load_from_disk)")
    parser.add_argument("--pretrained_model_name", type=str, default="gpt2",
                        help="Name or path of the pretrained model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--k", type=int, default=4, help="Number of implicit tokens")
    parser.add_argument("--wandb_project", type=str, default="implicit-tokens-alignment",
                        help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="run1",
                        help="wandb run name")
    args = parser.parse_args()

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    
    # Initialize wandb for logging.
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Initialize your custom model.
    model = ImplicitTokensAlignment(
        pretrained_model_name=args.pretrained_model_name,
        k=args.k,
        device="cuda"  # Trainer will handle device placement automatically.
    )

    # Load the dataset from disk.
    dataset = load_from_disk(args.dataset_path)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Dataset must contain both 'train' and 'test' splits.")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Preprocess the datasets.
    def preprocess_fn(example):
        return preprocess_example(example, model.tokenizer, system_prompt)
    train_dataset = train_dataset.map(preprocess_fn, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_fn, remove_columns=test_dataset.column_names)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",      # Evaluate at the end of each epoch.
        save_strategy="epoch",            # Save checkpoints at the end of each epoch.
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,      # Load the best model at the end of training.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"],              # Log metrics to wandb.
        run_name=args.wandb_run_name,
        save_total_limit=1,
        fp16=True,                        # Use mixed precision training.
        fp16_opt_level="O3",
        fp16_backend="apex",
        half_precision_backend="apex",
        # DeepSpeed integration:
        deepspeed="configs/zero++.json",  # Path to your DeepSpeed config file.
        # (Optional) Use Torch's compile mode for potential performance gains (requires PyTorch 2.0+):
        torch_compile=True,
        # Improve data loading efficiency:
        dataloader_num_workers=4,
        remove_unused_columns=False
    )

    # Create the Trainer instance.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=lambda batch: data_collator(batch, model.tokenizer),
        tokenizer=model.tokenizer,
        # Optionally, add compute_metrics function if you want additional metrics.
    )

    # Start training. The Trainer will handle evaluation, logging (to wandb), and saving the best model.
    trainer.train()

    # Run a final evaluation and print the results.
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    wandb.finish()
    print("Training complete. Best model saved.")

if __name__ == "__main__":
    main()