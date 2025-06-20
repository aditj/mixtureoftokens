import argparse
import os
import random
import re
import json
from typing import List, Tuple
import pdb
import torch
import torch.nn.functional as F
import evaluator
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
# -----------------------------------------------------------------------------
# Utility helpers (answer extraction, sampling, log-prob computation)
# -----------------------------------------------------------------------------

def parse_ground_truth(answer_str: str):
    """Parse the numerical answer provided by GSM8K."""
    text = answer_str.split("####")[-1].strip()
    text = text.replace(",", "").rstrip(". ")
    try:
        return float(text)
    except ValueError:
        if "/" in text and len(text.split("/")) == 2:
            num, den = text.split("/")
            try:
                return float(num.strip()) / float(den.strip())
            except ValueError:
                pass
    return None






# -----------------------------------------------------------------------------
# Embedding-mixture generation (two-phase) – simplified from simple.py
# -----------------------------------------------------------------------------
def generate_with_embedding_mixture(
    model,
    tokenizer,
    prompt_text: str,
    *,
    T_e: int = 50,
    T_exp: int = 200,
    k: int = 5,
    temperature: float = 0.6,
    min_end_prob: float = 0.5,
    max_prompt_length: int = 1024,
    num_chains: int = 1,
    max_completion_length: int = 1024,
    experiment_name: str = "non_uniform"
) -> str | tuple[str, dict]:
    
    device = model.device
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight
    
    # Encode prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    prompt_ids = prompt_ids[-max_prompt_length:].to(device)
    prompt_mask = prompt_mask[-max_prompt_length:].to(device)
    
    # Replicate for num_chains
    input_ids = prompt_ids.repeat(num_chains, 1).to(device)
    attention_mask = prompt_mask.repeat(num_chains, 1).to(device)
    
    # Get initial state
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past_key_values = outputs.past_key_values
    
    # Initialize tracking
    generated_token_ids = [[] for _ in range(num_chains)]
    
    # State tracking arrays
    active_chains = torch.ones(num_chains, dtype=torch.bool, device=device)  # True for mixture, False for sampling
    running_chains = torch.ones(num_chains, dtype=torch.bool, device=device)  # True if not stopped by EOS
    chain_lengths = torch.zeros(num_chains, dtype=torch.long, device=device)  # How long each chain ran
    
    # Find end token for stopping condition
    end_token_id = None
    for candidate in ['</', 'think']:
        try:
            end_token_id = tokenizer.encode(candidate, add_special_tokens=False)[0]
            break
        except:
            continue
    
    def obtain_probs_from_logits(logits, temp):
        """Helper function to get probabilities from logits with temperature"""
        scaled_logits = logits / temp
        return F.softmax(scaled_logits, dim=-1)
    # tokens for thinking
    think_start_token = "<think>"
    think_end_token = "</think>"
    # Add initial <think> token for all chains
    try:
        think_start_ids = tokenizer.encode(think_start_token, add_special_tokens=False)
        for token_id in think_start_ids:
            for chain_idx in range(num_chains):
                generated_token_ids[chain_idx].append(token_id)
            
            token_embeddings = embedding_layer(
                torch.tensor([[token_id] * num_chains], device=device).T
            ).to(embedding_matrix.dtype)
            
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=token_embeddings,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = outputs.past_key_values
    except:
        print("Warning: Could not add <think> token")
    
    # Main generation loop
    for t in range(T_e + T_exp):
        if not running_chains.any():
            break
            
        # Get probabilities from current logits
        last_logits = outputs.logits[:, -1, :]  # [num_chains, vocab_size]
        probs = obtain_probs_from_logits(last_logits, temperature)
        
        # Check for end token probability and update active status
        think_end_token_probs = probs[:, end_token_id]  # [num_chains]
        should_deactivate = (think_end_token_probs >= min_end_prob) & active_chains
        active_chains = active_chains & ~should_deactivate
        
        # Log deactivations
        for chain_idx in range(num_chains):
            if should_deactivate[chain_idx]:
                print(f"Stopping Phase 1 for chain {chain_idx} at round {t} due to end token probability: {think_end_token_probs[chain_idx]:.3f}")
        
    
        # Prepare embeddings for each chain
        embeddings_list = []
        
        for chain_idx in range(num_chains):
            if not running_chains[chain_idx]:
                # Chain already stopped, use zero embedding
                embeddings_list.append(torch.zeros(1, embedding_matrix.size(1), device=device, dtype=embedding_matrix.dtype))
                continue
                
            if active_chains[chain_idx]:
                # Active chain: use mixture embedding
                chain_probs = probs[chain_idx]
                top_k_probs, top_k_indices = torch.topk(chain_probs, k)
                if experiment_name == "non_uniform":
                    normalized_probs = top_k_probs / top_k_probs.sum()  
                else:
                    normalized_probs = torch.ones_like(top_k_probs) / k
                
                # Sample mixture weights from Dirichlet
                mixture_weights = torch.distributions.dirichlet.Dirichlet(normalized_probs).sample()
                mixture_weights = mixture_weights / mixture_weights.sum()
                
                # Get embeddings and compute mixture
                top_k_embeddings = embedding_matrix[top_k_indices]  # [k, hidden_size]
                mixed_embedding = torch.sum(top_k_embeddings * mixture_weights.unsqueeze(-1), dim=0)
                embeddings_list.append(mixed_embedding.unsqueeze(0))
                
                # Track token for logging (sample from normalized probs)
                sampled_idx = torch.multinomial(normalized_probs, num_samples=1).item()
                actual_token_id = top_k_indices[sampled_idx].item()
                generated_token_ids[chain_idx].append(actual_token_id)
                
            else:
                if should_deactivate[chain_idx]:
                    print(f"Add thinking end token in Phase 1 for chain {chain_idx} at round {t}")
                    think_end_ids = tokenizer.encode(think_end_token, add_special_tokens=False)
                    sampled_token_id = think_end_ids[0]
                else:
                        # Inactive chain: sample token and use its embedding
                    sampled_token_id = torch.multinomial(probs[chain_idx], num_samples=1).item()
                token_embedding = embedding_layer(torch.tensor([sampled_token_id], device=device))
                embeddings_list.append(token_embedding)
                
                generated_token_ids[chain_idx].append(sampled_token_id)
                
                # Check for EOS
                if sampled_token_id == tokenizer.eos_token_id:
                    running_chains[chain_idx] = False
                    chain_lengths[chain_idx] = t
        
        # Concatenate embeddings in correct order and feed to model
        token_embeddings_batch = torch.stack(embeddings_list, dim=0)  # [num_chains, 1, hidden_size]
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=token_embeddings_batch,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values
    
    # Update chain lengths for chains that didn't stop
    for chain_idx in range(num_chains):
        if running_chains[chain_idx]:
            chain_lengths[chain_idx] = T_e + T_exp
    
    # Process results for all chains (same as original)
    batched_prompt_completion_ids = []
    batched_prompt_ids = []
    batched_generated_ids = []
    batched_attention_masks = []
    batched_generated_text = []
    
    for chain_idx in range(num_chains):
        # Decode generated tokens for this chain
        generated_text = tokenizer.decode(generated_token_ids[chain_idx], skip_special_tokens=True)
        batched_generated_text.append(generated_text)
        
        # Tensorify the generated token ids
        chain_generated_ids = torch.tensor(generated_token_ids[chain_idx], device=device).reshape(1, -1)
        
        # Masking for this chain
        is_eos = chain_generated_ids == tokenizer.eos_token_id
        if is_eos.any():
            eos_idx = is_eos.int().argmax()
        else:
            eos_idx = chain_generated_ids.size(1) - 1
            
        sequence_indices = torch.arange(max_completion_length, device=device).reshape(1, -1)
        completion_mask = (sequence_indices <= eos_idx).int().to(device)
        
        # Use the original single prompt for each chain
        chain_prompt_mask = prompt_mask.unsqueeze(0) if prompt_mask.dim() == 1 else prompt_mask
        chain_prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        
        chain_attention_mask = torch.cat([chain_prompt_mask, completion_mask], dim=1)
        chain_prompt_completion_ids = torch.cat([chain_prompt_ids, chain_generated_ids], dim=1).to(device)
        
        # Pad to same length as attention mask
        if chain_prompt_completion_ids.size(1) < chain_attention_mask.size(1):
            padding_length = chain_attention_mask.size(1) - chain_prompt_completion_ids.size(1)
            padding = torch.full((1, padding_length), tokenizer.pad_token_id, device=device)
            chain_prompt_completion_ids = torch.cat([chain_prompt_completion_ids, padding], dim=1)
        
        # Collect tensors for batching
        batched_prompt_completion_ids.append(chain_prompt_completion_ids)
        batched_prompt_ids.append(chain_prompt_ids)
        batched_generated_ids.append(chain_generated_ids)
        batched_attention_masks.append(chain_attention_mask)
        
    
    # Find maximum sequence length for padding
    max_seq_len = max(tensor.size(1) for tensor in batched_prompt_completion_ids)
    max_prompt_len = max(tensor.size(1) for tensor in batched_prompt_ids)
    max_gen_len = max(tensor.size(1) for tensor in batched_generated_ids)
    max_attention_len = max(tensor.size(1) for tensor in batched_attention_masks)
    
    # Pad all tensors to the same length and batch them
    def pad_and_batch_tensors(tensor_list, max_len, pad_token_id=tokenizer.pad_token_id):
        padded_tensors = []
        for tensor in tensor_list:
            if tensor.size(1) < max_len:
                padding = torch.full((1, max_len - tensor.size(1)), pad_token_id, device=device)
                padded_tensor = torch.cat([tensor, padding], dim=1)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        return torch.cat(padded_tensors, dim=0)  # Stack along batch dimension
    
    # Batch all tensors
    final_prompt_completion_ids = pad_and_batch_tensors(batched_prompt_completion_ids, max_seq_len)
    final_prompt_ids = pad_and_batch_tensors(batched_prompt_ids, max_prompt_len)
    final_generated_ids = pad_and_batch_tensors(batched_generated_ids, max_gen_len)
    final_attention_mask = pad_and_batch_tensors(batched_attention_masks, max_attention_len, pad_token_id=0)
    
    # For variables that are the same across chains, copy them
    batched_prompt_text = [prompt_text] * num_chains
    
    return final_prompt_completion_ids, final_prompt_ids, final_generated_ids, final_attention_mask, batched_generated_text, batched_prompt_text

# -----------------------------------------------------------------------------
# Log-prob utilities for GRPO
# -----------------------------------------------------------------------------

def get_per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """Return log-probabilities for the completion tokens.

    Shapes:
        input_ids – (B, L)
        attention_mask – (B, L)
    Returns:
        logps – (B, L-1)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)
    logits = logits[:, -logits_to_keep:, :]
    labels = input_ids[:, -logits_to_keep:]

    # Handle potential inf/nan in logits (use safe value for float16)
    # logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -65000.0))
    
    logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logps


# -----------------------------------------------------------------------------
# GRPO loss (minimal version)
# -----------------------------------------------------------------------------
def score_completions(
    completions_text: list[str],
    question: str,
    answer: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    current_step: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
            'answer': answer
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    mock_prompts = [[{'content': question}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]
    answers = [answer] * len(completions_text)
    
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=answers,
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)
    ### store the generations in a txt file for each step
    os.makedirs(f"training_logs/run_{args.current_run}/logs", exist_ok=True)
    with open(f"training_logs/run_{args.current_run}/logs/generations_{current_step}.txt", "a") as f:
        for generation in log_data['generations']:
            f.write(f"--------------------------------\n")
            f.write(f"Question: {question}\n")
            f.write(f"{generation['response']}\n")
            f.write(f"{generation['scores']}\n")
            f.write("\n")

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }
    # pdb.set_trace()
    return rewards, advantages, rewards_per_func, metrics, log_data

def compute_loss(
    model: AutoModelForCausalLM,
    base_model: AutoModelForCausalLM, 
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_mask.size(1)
    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = get_per_token_logps(base_model, prompt_completion_ids, attention_mask,logits_to_keep)

    # Get training model logits
    input_ids = prompt_completion_ids#ß torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = get_per_token_logps(model, input_ids, attention_mask,logits_to_keep)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics

def grpo_loss(
        model: AutoModelForCausalLM,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        question: str,
        answer: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        arguments: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """
    # Generate completions
    prompt_text = create_prompt(tokenizer, question)
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_with_embedding_mixture(
            model, tokenizer, prompt_text, num_chains=arguments.num_chains, T_e=arguments.T_e, T_exp= int(arguments.T - arguments.T_e), 
            temperature=arguments.temperature,k=arguments.k, experiment_name=arguments.experiment_name
        )
    arguments.kl_weight_beta = 0.02
    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, arguments, round_num
    )

    # Compute loss
    

    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        model, base_model, prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, arguments
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics, rewards, advantages, rewards_per_func, log_data

# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def create_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    pre_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <think> tags and your final answer inside <answer> tags, like this:
            <think>
            Your step-by-step reasoning process here
            </think>
            <answer>
            Your final answer here
            </answer>"""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

# -----------------------------------------------------------------------------
# Main training / evaluation script
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Embedding-mixture GRPO demo")
    ###### Arguments for model ######
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.6)

    ###### Arguments for training ######
    parser.add_argument("--steps", type=int, default=1000, help="Number of GRPO updates (demo)")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    ###### Arguments for GRPO ######
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--warmup_percent", type=float, default=0.18)
    parser.add_argument("--num_chains", type=int, default=10)
    ###### Arguments for updating reference model ######
    parser.add_argument("--update_ref_model", action="store_true")
    parser.add_argument("--update_ref_model_freq", type=int, default=200)
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1)
    ###### Arguments for embedding mixture ######
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--T_e", type=int, default=400)
    parser.add_argument("--T",type=int,default=1000)
    parser.add_argument("--experiment_name",type=str,default="non_uniform")
    parser.add_argument("--slurm_id",type=int,default=0)

    # wandb arguments
    parser.add_argument("--wandb_project", type=str, default="grpo-gsm8k-mix")
    parser.add_argument("--wandb_entity", type=str, default="aditjain1980-cornell-university")
    parser.add_argument("--wandb_log", action="store_true", default=True, help="Log to wandb")
    #### get last run 
    args = parser.parse_args()
    os.makedirs("training_logs", exist_ok=True)
    last_run = sorted(os.listdir("training_logs"))[-1]
    current_run = int(last_run.split("_")[-1])+1 + args.slurm_id
    

    args.current_run = current_run
    args.wandb_run_name = f"grpo-gsm8k-mix_{current_run}_model_{args.model.split('/')[-1]}_k_{args.k}_T_e_{args.T_e}_T_exp_{args.T - args.T_e}"+f"_expname_{args.experiment_name}"
    
    return args

def main():
    args = parse_args()
    WANDB_LOG = args.wandb_log

    # Initialize wandb
    if WANDB_LOG:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
            tags=["grpo", "embedding-mixture", args.model.split("/")[-1]]
        )
        
        # Log additional config
        wandb.config.update({
            "T_exp": args.T - args.T_e,
        })
    
    T_e = args.T_e
    T_exp = args.T - T_e
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.train()
    

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        eps=1e-8
    )

    warmup_steps = int(args.warmup_percent * args.steps)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)


    # small subset of GSM8K train split
    ds = load_dataset("gsm8k", "main", split="train[:10%]")
    accumulated_loss = 0
    optimizer.zero_grad()
    
    # Create training logs directory
    os.makedirs("training_logs", exist_ok=True)
    
    for step in tqdm(range(args.steps), desc="Training Progress"):
        sample = ds[random.randint(0, len(ds) - 1)]
        question = sample["question"]
        gt_val = parse_ground_truth(sample["answer"])
        
        if args.update_ref_model and (step+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data


        # compute reward
        loss, metrics, rewards, advantages, rewards_per_func, log_data = grpo_loss(
            model,
            ref_model,
            tokenizer,
            question,
            gt_val,
            evaluator.GSM8kEvaluator(),
            args.device,
            step,
            "training_logs",
            args
        )
        print(f"--------------------------------")
        print(f"loss: {loss}    rewards: {rewards} advantages: {advantages} rewards_per_func: {rewards_per_func}")
        print(f"--------------------------------")

        
        # Gradient accumulation
        total_loss = loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += loss.item()   
        scheduler.step()

        # Step optimizer
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    
            accumulated_loss = 0

        print(f"Step {step:3d} | Loss: {loss.item():.4f} | GT: {gt_val}")
        torch.cuda.empty_cache()
        # Log to wandb
        if WANDB_LOG:
            wandb_log = {
                "train/loss": loss.item(),
                "train/step": step,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/gt_value": gt_val,
                **{f"train/{k}": v for k, v in metrics.items()},
                "train/mean_reward": rewards.mean().item(),
                "train/std_reward": rewards.std().item(),
                "train/max_reward": rewards.max().item(),
                "train/min_reward": rewards.min().item(),
                "train/mean_advantage": advantages.mean().item(),
                "train/std_advantage": advantages.std().item(),
            }
            
            # Log individual reward components if available
            if rewards_per_func is not None and len(rewards_per_func.shape) > 1:
                for i in range(rewards_per_func.shape[1]):
                    wandb_log[f"train/reward_component_{i}"] = rewards_per_func[:, i].mean().item()
            
            # Log sample generation every 50 steps
            if step % 50 == 0 and len(log_data['generations']) > 0:
                # Create a table with generations
                generation_data = []
                for i, gen in enumerate(log_data['generations'][:3]):  # Log first 3 generations
                    generation_data.append([
                        step,
                        i,
                        question[:100] + "..." if len(question) > 100 else question,
                        gen['response'][:200] + "..." if len(gen['response']) > 200 else gen['response'],
                        gen['scores']['total_reward'],
                        gt_val
                    ])
                
                generations_table = wandb.Table(
                    columns=["step", "chain_id", "question", "response", "reward", "ground_truth"],
                    data=generation_data
                )
                wandb_log["train/generations"] = generations_table
            
            wandb.log(wandb_log, step=step)
        
        ##### log data #####
        with open(f"training_logs/run_{args.current_run}/log_data.json", "a") as f:
            json.dump(log_data, f)
            f.write('\n')  # Add newline for easier reading
    ### save model ####
    torch.save(model.state_dict(), f"training_logs/run_{args.current_run}/model.pt")

if __name__ == "__main__":
    main() 