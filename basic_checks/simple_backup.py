### simple script to test the Qwen3 model on Gsm8k test set on 200 examples and see the pass@1 accuracy 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import json
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import pickle
import numpy as np
import argparse

def extract_answer(text):
    """Extract numerical answer from generated text."""
    # Look for boxed answers first (from the system prompt)
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(boxed_pattern, text)
    if match:
        try:
            # Handle fractions and basic expressions
            answer_str = match.group(1).strip()
            if '/' in answer_str and len(answer_str.split('/')) == 2:
                num, den = answer_str.split('/')
                return float(num.strip()) / float(den.strip())
            return float(answer_str)
        except ValueError:
            pass
    
    # Look for patterns like "#### 42" or "The answer is 42"
    patterns = [
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"(?:the )?answer is:?\s*(-?\d+(?:\.\d+)?)",
        r"(?:therefore|so|thus),?\s*(?:the )?answer is:?\s*(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Fallback: look for the last number in the text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def parse_ground_truth(answer_str: str):
    """Parse ground truth answer from GSM8K format, handling commas and fractions."""
    # Remove the explanatory solution before the delimiter if present
    text = answer_str.split("####")[-1].strip()
    # Remove thousands separators
    text = text.replace(",", "")
    # Strip trailing punctuation
    text = text.strip().rstrip(".")
    
    # Try simple float conversion first
    try:
        return float(text)
    except ValueError:
        # Handle simple fractions a/b
        if '/' in text and len(text.split('/')) == 2:
            num, den = text.split('/')
            try:
                return float(num.strip()) / float(den.strip())
            except ValueError:
                pass
    
    # Could not parse
    return None

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


def generate_in_embedding_space(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    stop_token_id: int | None = None,
    return_token_ids: bool = False,
) -> str | tuple[str, list[int]]:
    """Generate text *token-by-token* by manually feeding the **embeddings** of the sampled
    tokens back into the model.

    This mirrors what :pymeth:`~transformers.AutoModelForCausalLM.generate` does internally
    but exposes the embedding-space interface explicitly, which can be useful for research
    experiments.

    The function operates on a *single* prompt string for clarity. It returns **only** the
    *newly* generated text (without the prompt).
    """

    device = model.device

    # ---------------------------------------------------------------------
    # Encode the prompt and run a forward pass to prime the KV cache
    # ---------------------------------------------------------------------
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)

    # Retrieve past_key_values to avoid re-computing the prompt context each step
    past_key_values = outputs.past_key_values

    # We will sequentially append generated token ids here
    generated_token_ids: list[int] = []

    # We'll repeatedly sample tokens until we reach ``max_new_tokens`` or hit ``stop_token_id``
    for _ in range(max_new_tokens):
        # -----------------------------------------------------------------
        # Sample next token from the logits of the *last* position
        # -----------------------------------------------------------------
        last_logits = outputs.logits[:, -1, :].squeeze(0)  # -> [vocab_size]
        next_token_id = _sample_token_from_logits(
            last_logits, temperature=temperature, top_p=top_p
        )
        generated_token_ids.append(next_token_id)

        # Early stopping if EOS or custom stop token is generated
        if (stop_token_id is not None and next_token_id == stop_token_id) or (
            stop_token_id is None and next_token_id == tokenizer.eos_token_id
        ):
            break

        # -----------------------------------------------------------------
        # Convert the sampled token id to its **embedding** and feed it back
        # -----------------------------------------------------------------
        embedding_layer = model.get_input_embeddings()
        next_token_embed = embedding_layer(
            torch.tensor([[next_token_id]], device=device)
        )  # Shape: [1,1,hidden_size]

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        # Update KV cache for the next iteration
        past_key_values = outputs.past_key_values

    # ---------------------------------------------------------------------
    # Decode only the newly generated ids (excluding the prompt)
    # ---------------------------------------------------------------------
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    if return_token_ids:
        return generated_text, generated_token_ids
    return generated_text

def generate_with_embedding_mixture(
    model,
    tokenizer,
    prompt: str,
    *,
    T_e: int = 50,
    T_exp: int = 200,
    k: int = 5,
    temperature: float = 0.8,
    min_end_prob: float = 0.1,
    return_phase_info: bool = False,
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
    
    # Phase tracking
    phase1_tokens = []
    phase2_tokens = []
    transition_tokens = []
    phase1_rounds_completed = 0
    
    # Try to find '</' token id for stopping condition
    end_token_candidates = ['</', '</']
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
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        
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
        normalized_probs = top_k_probs / top_k_probs.sum()
        
        # Weighted sum: [k, hidden_size] * [k, 1] -> [hidden_size]
        mixed_embedding = torch.sum(
            top_k_embeddings * normalized_probs.unsqueeze(-1), 
            dim=0
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        
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
        generated_token_ids.append(actual_token_id)
        phase1_tokens.append(actual_token_id)
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
    # Add <think> token to start phase 2
    # -------------------------------------------------------------------------
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
    print(f"Phase 2: Standard generation for {T_exp} rounds...")
    
    for _ in range(T_exp):
        # Standard sampling from logits
        last_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token_id = _sample_token_from_logits(last_logits, temperature=temperature)
        
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
    args = parser.parse_args()

    T_e = args.T_e
    k = args.k
    T_total = args.T_total
    T_exp = T_total - T_e
    num_examples = args.num_examples
    temperature = args.temperature

    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # Using Qwen2.5 as it's more readily available
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    
    # Take first 200 examples
    examples = test_data.select(range(num_examples))
    
    # Extract questions and ground truth answers
    questions = [example["question"] for example in examples]
    ground_truths = [parse_ground_truth(example["answer"]) for example in examples]
    
    # Set batch size (adjust based on your GPU memory)
    
    total_batches = (len(questions) + batch_size - 1) // batch_size
    
    print(f"Running inference on {num_examples} examples with batch size {batch_size}...")
    print(f"Total batches: {total_batches}")
    
    
    # ---------------------------------------
    # 1) Standard embedding-space generation
    # ---------------------------------------
    
    # ---------------------------------------
    # 2) Embedding mixture generation
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
                        "content": "You are Qwen. You are a helpful mathematical reasoning assistant. Think step by step and put your final answer within \\boxed{}.",
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
    
    # Save as JSON for human readability
    with open('generation_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print("Detailed comparison data saved to 'generation_comparison.json'")
    
    # Save as pickle for fast loading
    with open('generation_comparison.pkl', 'wb') as f:
        pickle.dump(comparison_data, f)
    print("Raw comparison data saved to 'generation_comparison.pkl'")


if __name__ == "__main__":
    main()

