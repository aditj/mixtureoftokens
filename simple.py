import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets_custom.gsm8k import extract_answer, parse_ground_truth
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import numpy as np
import argparse
import os
import json 
# -----------------------------------------------------------------------------
# Embedding-space generation utilities
# -----------------------------------------------------------------------------

def _sample_token_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """Sample a single token ID from the given logits using temperature and top-p (nucleus) sampling.

    Args:
        logits (torch.Tensor): Logits tensor of shape ``[vocab_size]``.
        temperature (float): Temperature value for scaling the logits.
        top_p (float): Cumulative probability for nucleus sampling. If set to ``1.0`` no
            nucleus filtering is applied.

    Returns:
        int: Sampled token id.
    """
    # Temperature scaling
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep only tokens with cumulative probability <= top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Ensure at least one token is kept
        sorted_indices_to_remove[..., 0] = False

        probs_to_keep = sorted_probs.clone()
        probs_to_keep[sorted_indices_to_remove] = 0.0

        # Scatter back to original indexing
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(dim=-1, index=sorted_indices, src=probs_to_keep)

        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    # Multinomial sampling to obtain the next token
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id.item()

def generate_with_embedding_mixture(
    model,
    tokenizer,
    prompt: str,
    *,
    T_e: int = 50,
    T_exp: int = 200,
    k: int = 5,
    top_p: float = 0.95,
    temperature: float = 0.8,
    min_end_prob: float = 0.1,
    return_phase_info: bool = False,
    PHASE_2_STRATEGY: str = "think_first",  
    experiment_name: str = "non_uniform",
) -> str | tuple[str, dict]:
    """Generate text using a two-phase approach:
    
    Phase 1 (T_e rounds): Use weighted mixture of top-k token embeddings
    Phase 2 (T_exp rounds): Standard token-by-token generation after </think>
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        T_e: Number of embedding mixture rounds
        T_exp: Number of standard generation rounds after </think>
        k: Number of top tokens to consider for mixture
        temperature: Temperature for sampling
        min_end_prob: Minimum probability for '</' token to stop phase 1
        return_phase_info: Whether to return detailed phase information
    
    Returns:
        Generated text (only the new part, without prompt), optionally with phase info
    """
    device = model.device
    
    # Get embedding matrix for efficient lookup
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight  # Shape: [vocab_size, hidden_size]
    
    # Encode prompt and get initial state
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    
    past_key_values = outputs.past_key_values
    generated_token_ids = []
    generated_token_ids_all = []
    # Phase tracking
    phase1_tokens = []
    phase2_tokens = []
    transition_tokens = []
    phase1_rounds_completed = 0
    # Try to find '</' token id for stopping condition
    end_token_candidates = ['</', 'think']
    end_token_id = None
    for candidate in end_token_candidates:
        try:
            end_token_id = tokenizer.encode(candidate, add_special_tokens=False)[0]
            break
        except:
            continue
    
    # -------------------------------------------------------------------------
    # Phase 1: Embedding mixture generation for T_e rounds
    # -------------------------------------------------------------------------
    print(f"Phase 1: Embedding mixture generation for {T_e} rounds...")
    
    for round_idx in range(T_e):
        # Get logits from current state
        last_logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab_size]
        
        # Apply temperature scaling
        scaled_logits = last_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Get top-k tokens and their probabilities
        if "nucleus" in experiment_name:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            top_k_probs, top_k_indices = sorted_probs[:torch.searchsorted(cum_probs, top_p)+1], sorted_indices[:torch.searchsorted(cum_probs, top_p)+1]
        else:
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        if probs[end_token_id] > 0.1:
            print(f"Stopping Phase 1 at round {round_idx} due to end token probability: {probs[end_token_id]:.3f}")
            break
        # Check stopping condition: if '</' token has minimum probability
        if end_token_id is not None and end_token_id in top_k_indices:
            end_token_pos = (top_k_indices == end_token_id).nonzero(as_tuple=True)[0]
            if len(end_token_pos) > 0:
                end_prob = top_k_probs[end_token_pos[0]]
                if end_prob >= min_end_prob:
                    print(f"Stopping Phase 1 at round {round_idx} due to end token probability: {end_prob:.3f}")
                    break
        
        # Get embeddings for top-k tokens
        top_k_embeddings = embedding_matrix[top_k_indices]  # [k, hidden_size]
        
        # Create weighted mixture of embeddings
        # Normalize probabilities to sum to 1
        if "element_wise_max" in experiment_name:
            ### the mixed embedding is the element-wise max of the top-k embeddings
            mixed_embedding = torch.max(top_k_embeddings, dim=0).values        
        else:
            if "inverse_p" in experiment_name:
                ### weighted mixture
                normalized_probs = 1 / top_k_probs
            else:
                ### weighted mixture
                normalized_probs = top_k_probs / top_k_probs.sum()
                
            # Weighted sum: [k, hidden_size] * [k, 1] -> [hidden_size]
            mixed_embedding = torch.sum(
                top_k_embeddings * normalized_probs.unsqueeze(-1), 
                dim=0
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        if "dirichlet" in experiment_name:
            ### take a dirichlet sample with the same mean as the top-k embeddings
            normalized_probs_dirichlet = normalized_probs.clone() + 1e-6
            normalized_probs_dirichlet = normalized_probs_dirichlet / normalized_probs_dirichlet.sum()
            d = torch.distributions.dirichlet.Dirichlet(normalized_probs_dirichlet)
            sampling_probs = d.sample()
            mixed_embedding = torch.sum(
                top_k_embeddings * sampling_probs.unsqueeze(-1), 
                dim=0
            ).unsqueeze(0).unsqueeze(0)

        # Feed mixed embedding back to model
        with torch.no_grad():
            outputs = model(
                inputs_embeds=mixed_embedding,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        
        # For tracking purposes, sample one token from the mixture to add to generated_token_ids
        sampled_token_id = torch.multinomial(normalized_probs, num_samples=1).item()
        actual_token_id = top_k_indices[sampled_token_id].item()
        ### store all the top-k tokens along with their probabilities
        generated_token_ids.append(actual_token_id)
        phase1_tokens.append((top_k_indices.tolist(), normalized_probs.tolist()))
        phase1_rounds_completed += 1
    
    # -------------------------------------------------------------------------
    # Add </think> token
    # -------------------------------------------------------------------------
    think_end_token = "</think>"
    try:
        think_end_ids = tokenizer.encode(think_end_token, add_special_tokens=False)
        for token_id in think_end_ids:
            generated_token_ids.append(token_id)
            transition_tokens.append(token_id)
            
            # Feed the actual token embedding
            token_embedding = embedding_layer(
                torch.tensor([[token_id]], device=device)
            )
            
            with torch.no_grad():
                outputs = model(
                    inputs_embeds=token_embedding,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = outputs.past_key_values
    except:
        print("Warning: Could not add </think> token")
    
    # -------------------------------------------------------------------------
    # Set temperature for phase 2 to 0.6 for more focused generation
    # -------------------------------------------------------------------------
    phase2_temperature = 0.6
    print(f"Setting temperature for Phase 2 to {phase2_temperature}")
    
    # -------------------------------------------------------------------------
    # Add <think> token to start phase 2
    # -------------------------------------------------------------------------
    if PHASE_2_STRATEGY == "think_first":
        think_start_token = "<think>"
        try:
            think_start_ids = tokenizer.encode(think_start_token, add_special_tokens=False)
            for token_id in think_start_ids:
                generated_token_ids.append(token_id)
                transition_tokens.append(token_id)
                
                # Feed the actual token embedding
                token_embedding = embedding_layer(
                    torch.tensor([[token_id]], device=device)
                )
                
                with torch.no_grad():
                    outputs = model(
                        inputs_embeds=token_embedding,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = outputs.past_key_values
        except:
            print("Warning: Could not add <think> token")
    
    # -------------------------------------------------------------------------
    # Phase 2: Standard token-by-token generation for T_exp rounds
    # -------------------------------------------------------------------------
    if PHASE_2_STRATEGY == "answer_first":
        answer_start_token = "<answer>"
        try:
            answer_start_ids = tokenizer.encode(answer_start_token, add_special_tokens=False)
            for token_id in answer_start_ids:
                generated_token_ids.append(token_id)
                transition_tokens.append(token_id)
                
                # Feed the actual token embedding
                token_embedding = embedding_layer(
                    torch.tensor([[token_id]], device=device)
                )
                
                with torch.no_grad():
                    outputs = model(
                        inputs_embeds=token_embedding,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = outputs.past_key_values
        except:
            print("Warning: Could not add <answer> token")
    print(f"Phase 2: Standard generation for {T_exp} rounds...")
    
    for _ in range(T_exp):
        # Standard sampling from logits
        last_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token_id = _sample_token_from_logits(last_logits, temperature=phase2_temperature)
        
        generated_token_ids.append(next_token_id)
        phase2_tokens.append(next_token_id)
        
        # Early stopping on EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Feed token embedding back
        token_embedding = embedding_layer(
            torch.tensor([[next_token_id]], device=device)
        )
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=token_embedding,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,

            )
        
        past_key_values = outputs.past_key_values
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    if return_phase_info:
        phase_info = {
            'total_tokens': len(generated_token_ids),
            'phase1_tokens': len(phase1_tokens),
            'phase2_tokens': len(phase2_tokens),
            'transition_tokens': len(transition_tokens),
            'phase1_rounds_completed': phase1_rounds_completed,
            'phase1_rounds_requested': T_e,
            'phase2_rounds_requested': T_exp,
            'phase1_token_ids': phase1_tokens,
            'phase2_token_ids': phase2_tokens,
            'transition_token_ids': transition_tokens,
        }
        return generated_text, phase_info
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Compare standard and embedding mixture generation on GSM8K.")
    parser.add_argument('--T_e', type=int, default=200, help='Number of embedding mixture rounds (phase 1)')
    parser.add_argument('--k', type=int, default=5, help='Top-k tokens for mixture in phase 1')
    parser.add_argument('--T_total', type=int, default=600, help='Total number of generation steps for mixture method')
    parser.add_argument('--num_examples', type=int, default=50, help='Number of GSM8K examples to evaluate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--experiment_name', type=str, default="answer_directly_element_wise_max", help='Experiment name')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-3B-Instruct", help='Model name')
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='Load from checkpoint')
    args = parser.parse_args()

    T_e = args.T_e
    k = args.k
    T_total = args.T_total
    T_exp = T_total - T_e
    num_examples = args.num_examples
    temperature = args.temperature
    experiment_name = args.experiment_name
    print("Loading model and tokenizer...")
    model_name = args.model_name  # Using Qwen2.5 as it's more readily available
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if args.load_from_checkpoint is not None:
        model.load_state_dict(torch.load(args.load_from_checkpoint))
        model.to(torch.bfloat16)
        model.eval()
        print(f"Loaded model from checkpoint {args.load_from_checkpoint}")
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    
    # Take first 200 examples
    all_examples = test_data.shuffle(seed=42)
    examples = all_examples.select(range(num_examples))
    
    # Extract questions and ground truth answers
    questions = [example["question"] for example in examples]
    ground_truths = [parse_ground_truth(example["answer"]) for example in examples]

    if "answer_directly" in experiment_name:
        PHASE_2_STRATEGY = "answer_first"
    else:
        PHASE_2_STRATEGY = "think_first"

    # ---------------------------------------
    # 1) Embedding mixture generation
    # ---------------------------------------
    correct_mix = 0
    total_mix = 0
    mix_token_counts = []
    mix_results = []
    mix_phase_data = []
    with tqdm(total=len(questions), desc="Embedding mixture gen", unit="q") as pbar:
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            # Build the chat template prompt
            prompt_text = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": f"You are Qwen. You are a helpful mathematical reasoning assistant. Think step by step."+ "Put your final answer within \\boxed{}.",
                    },
                    {
                        "role": "user",
                        "content": f"<question> {question} </question> <think>",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate via embedding mixture approach with phase tracking
            answer_text, phase_info = generate_with_embedding_mixture(
                model,
                tokenizer,
                prompt_text,
                T_e=T_e,  # 100 rounds of embedding mixture
                T_exp=T_exp,  # 500 rounds of standard generation
                k=k,  # Top-5 tokens for mixture
                temperature=temperature,
                min_end_prob=0.05,  # Stop if '</' has 5% probability
                return_phase_info=True,
            )
            
            token_count = phase_info['total_tokens']
            mix_token_counts.append(token_count)
            mix_phase_data.append(phase_info)

            predicted_answer = extract_answer(answer_text)
            is_correct = predicted_answer is not None and abs(predicted_answer - ground_truth) < 1e-6

            if is_correct:
                correct_mix += 1
                
            # Store detailed results
            mix_results.append({
                'problem_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'token_count': token_count,
                'phase_info': phase_info,
                'answer_text': answer_text[:200] + '...' if len(answer_text) > 200 else answer_text
            })

            total_mix += 1

            pbar.update(1)
            pbar.set_postfix(
                {
                    "acc": f"{correct_mix/total_mix:.3f}",
                    "corr": f"{correct_mix}/{total_mix}",
                    "avg_tokens": f"{np.mean(mix_token_counts):.1f}",
                    "p1_avg": f"{np.mean([p['phase1_tokens'] for p in mix_phase_data]):.1f}",
                    "p2_avg": f"{np.mean([p['phase2_tokens'] for p in mix_phase_data]):.1f}",
                }
            )

    mix_accuracy = correct_mix / total_mix if total_mix else 0.0
    
    # ---------------------
    # Comparison summary
    # ---------------------
    print("\n\n==================== Summary ====================")
    print(f"Embedding mixture gen: {correct_mix}/{total_mix}  (Pass@1 = {mix_accuracy:.3f})")
    print(f"T_e: {T_e}, k: {k}, T_total: {T_total}, num_examples: {num_examples}, temperature: {temperature}")
    # ---------------------
    # Token Length Analysis
    # ---------------------
    print("\n\n==================== Token Length Analysis ====================")

    # Embedding mixture generation stats
    mix_avg_tokens = np.mean(mix_token_counts)
    mix_median_tokens = np.median(mix_token_counts)
    mix_std_tokens = np.std(mix_token_counts)
    
    # Phase-specific stats
    phase1_tokens = [p['phase1_tokens'] for p in mix_phase_data]
    phase2_tokens = [p['phase2_tokens'] for p in mix_phase_data]
    transition_tokens = [p['transition_tokens'] for p in mix_phase_data]
    phase1_rounds = [p['phase1_rounds_completed'] for p in mix_phase_data]
    
    print(f"\nEmbedding Mixture Generation:")
    print(f"  Total average tokens: {mix_avg_tokens:.1f}")
    print(f"  Total median tokens: {mix_median_tokens:.1f}")
    print(f"  Total std deviation: {mix_std_tokens:.1f}")
    print(f"  Total min tokens: {min(mix_token_counts)}")
    print(f"  Total max tokens: {max(mix_token_counts)}")
    
    print(f"\n  Phase 1 (Embedding Mixture):")
    print(f"    Average tokens: {np.mean(phase1_tokens):.1f}")
    print(f"    Average rounds completed: {np.mean(phase1_rounds):.1f} / 100 requested")
    print(f"    Min tokens: {min(phase1_tokens)}")
    print(f"    Max tokens: {max(phase1_tokens)}")
    
    print(f"\n  Phase 2 (Standard Generation):")
    print(f"    Average tokens: {np.mean(phase2_tokens):.1f}")
    print(f"    Min tokens: {min(phase2_tokens)}")
    print(f"    Max tokens: {max(phase2_tokens)}")
    
    print(f"\n  Transition tokens:")
    print(f"    Average tokens: {np.mean(transition_tokens):.1f}")
    
    # Comparison
    print(f"\n  Token Efficiency Comparison:")
  
    print(f"    Mixture approach uses {mix_avg_tokens:.1f} tokens on average")

    # ---------------------
    # Save detailed data
    # ---------------------
    print("\n\n==================== Saving Data ====================")
    
    comparison_data = {
        'metadata': {
            'num_examples': num_examples,
            'temperature': temperature,
            'model_name': model_name,
        },

        'embedding_mixture': {
            'accuracy': mix_accuracy,
            'correct': correct_mix,
            'total': total_mix,
            'token_counts': mix_token_counts,
            'results': mix_results,
            'phase_data': mix_phase_data,
            'stats': {
                'avg_tokens': float(mix_avg_tokens),
                'median_tokens': float(mix_median_tokens),
                'std_tokens': float(mix_std_tokens),
                'min_tokens': int(min(mix_token_counts)),
                'max_tokens': int(max(mix_token_counts)),
                'phase1_avg_tokens': float(np.mean(phase1_tokens)),
                'phase2_avg_tokens': float(np.mean(phase2_tokens)),
                'transition_avg_tokens': float(np.mean(transition_tokens)),
                'phase1_avg_rounds': float(np.mean(phase1_rounds)),
            }
        }
    }
    
    model_name = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    model_name += f"_{args.load_from_checkpoint}" if args.load_from_checkpoint is not None else ""
    
    os.makedirs(f'generation_comparison/{experiment_name}/{model_name}/{T_total}_{num_examples}_{temperature}', exist_ok=True)
    # Save as JSON for human readability
    with open(f'generation_comparison/{experiment_name}/{model_name}/{T_total}_{num_examples}_{temperature}/generation_comparison_T_e_{T_e}_k_{k}.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Detailed comparison data saved to 'generation_comparison/{experiment_name}/{model_name}/{T_total}_{num_examples}_{temperature}/generation_comparison_T_e_{T_e}_k_{k}.json'")
   

if __name__ == "__main__":
    main()

