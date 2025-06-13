import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from tqdm import tqdm
import time
from datasets import load_dataset
from datetime import datetime
import os
### library for arguments
import argparse
# Set random seed for reproducibility
torch.manual_seed(42)

print(f"Using CUDA: {torch.cuda.is_available()}")
print(f"Available GPUs: {torch.cuda.device_count()}")

def get_prob_for_next_token(model, tokenizer, current_tokens):
    inputs = tokenizer(current_tokens, return_tensors="pt")
    embedding_device = next(model.get_input_embeddings().parameters()).device
    inputs = {k: v.to(embedding_device) for k, v in inputs.items()}
    
    with torch.no_grad():  # Save memory by not storing gradients
        outputs = model(**inputs, max_new_tokens=1)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    return probs

def filter_probabilities(probs, top_k=30, top_p=0.95, min_non_zero_indices=8):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    sorted_probs = sorted_probs[:top_k]
    sorted_indices = sorted_indices[:top_k]
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    mask = cumsum_probs <= top_p
    indices_to_keep = sorted_indices[mask]
    if len(indices_to_keep) < min_non_zero_indices:
        indices_to_keep = sorted_indices[:min_non_zero_indices]
    
    probs_return = torch.zeros_like(probs, dtype=torch.half)
    probs_return[indices_to_keep] = probs[indices_to_keep]
    return probs_return

def get_embedding_for_token(model, tokenizer, token):
    inputs = tokenizer(token, return_tensors="pt")
    embedding_layer = model.get_input_embeddings()
    embedding_device = next(embedding_layer.parameters()).device
    inputs = {k: v.to(embedding_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embedding = embedding_layer(inputs['input_ids'])
    return embedding

def get_weighted_average_embedding(model, tokenizer, tokens, weights):
    embeddings = []
    for token_id, weight in zip(tokens, weights):
        token = tokenizer.decode(token_id)
        embedding = get_embedding_for_token(model, tokenizer, token)
        
        if isinstance(weight, torch.Tensor):
            weight = weight.to(embedding.device)
        else:
            weight = torch.tensor(float(weight), device=embedding.device)
        
        weighted_embedding = embedding * weight
        weighted_embedding = weighted_embedding.squeeze(0)
        embeddings.append(weighted_embedding)
    
    embeddings = torch.stack(embeddings)
    return torch.mean(embeddings, dim=0)

def get_next_token_using_embeddings(model, tokenizer, embeddings, num_samples=1, tau=0.2, temperature=0.6):
    # Add batch dimension if not present
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(0)
    
    seq_len = embeddings.shape[1]
    attention_mask = torch.ones(1, seq_len, device=embeddings.device)
    
    with torch.no_grad():
        transformer_outputs = model.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state
        logits = model.lm_head(hidden_states)
    
    logits = logits[0, -1, :] / temperature
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    
    entropy_less_than_tau = entropy < tau
    probs = filter_probabilities(probs)
    probs = probs / probs.sum()
    
    next_token = torch.multinomial(probs, num_samples, replacement=False)
    weights = probs[next_token.flatten().cpu().tolist()]
    weights = weights.clone().to(next_token.device)
    weights = weights / weights.sum()
    
    return next_token, entropy.item(), entropy_less_than_tau, weights

def extract_numeric_answer(answer_text):
    """Extract numeric answer from the generated text"""
    try:
        answer_match = re.search(r'<answer>\s*([+-]?\d+(?:\.\d+)?)\s*</answer>', answer_text)
        if answer_match:
            return float(answer_match.group(1))
        
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', answer_text)
        if numbers:
            return float(numbers[-1])
        
        return None
    except:
        return None

def process_single_question(question, ground_truth_answer, tokenizer, model, 
                          num_paths=4, num_samples=500, tau_threshold=0.00001, 
                          temperature=1.0, verbose=False, step_limit=400):
    """Process a single GSM8K question and return results"""
    
    # Format the question
    formatted_question = "You are tasked to solve the following math question. Please reason step by step, and put your final answer within \\boxed{}. The question is in <question> tags. You think in <think> tags and answer in <answer> tags. For example, if the answer is 8, you should answer <answer>\\boxed{8}</answer>"+ f"<question> {question} </question> <think>"
    messages = [{"role": "user", "content": formatted_question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    weights = torch.ones(num_paths) / num_paths
    entropies = []
    
    # Get embeddings for the input
    prev_input_tokens = tokenizer(text, return_tensors="pt")
    embedding_device = next(model.get_input_embeddings().parameters()).device
    prev_input_tokens = {k: v.to(embedding_device) for k, v in prev_input_tokens.items()}
    
    with torch.no_grad():
        prev_input_embedding = model.get_input_embeddings()(prev_input_tokens['input_ids'])
    
    prev_input_embedding = prev_input_embedding.squeeze(0)
    embeddings = torch.stack([prev_input_embedding])
    
    reasoning_tokens = []
    
    # Thinking phase
    for step in range(num_samples):
        next_tokens, entropy, entropy_less_than_tau, weights = get_next_token_using_embeddings(
            model, tokenizer, embeddings, num_paths, tau=tau_threshold, temperature=temperature
        )
        
        reasoning_tokens.append(tokenizer.decode(next_tokens))
        
        if "<|endoftext|>" in tokenizer.decode(next_tokens):
            break
        if entropy_less_than_tau or step > step_limit:
            if verbose:
                print(f"Stopping thinking phase at step {step} (entropy: {entropy:.6f})")
            break
        
        avg_embedding = get_weighted_average_embedding(model, tokenizer, next_tokens, weights)
        avg_embedding = avg_embedding.to(embeddings.device)
        embeddings = torch.cat([embeddings, avg_embedding.unsqueeze(0)], dim=1)
        entropies.append(float(entropy))
        
    # Add </think> token
    think_end_embedding = get_embedding_for_token(model, tokenizer, "</think>")
    think_end_embedding = think_end_embedding.to(embeddings.device)
    embeddings = torch.cat([embeddings, think_end_embedding], dim=1)
    
    # Answer phase
    answer_text = ""
    for step in range(50):
        next_tokens, entropy, entropy_less_than_tau, _ = get_next_token_using_embeddings(
            model, tokenizer, embeddings, 1, tau=tau_threshold, temperature=temperature
        )
        
        decoded_token = tokenizer.decode(next_tokens)
        if "<|endoftext|>" in decoded_token or "</answer>" in decoded_token:
            break
            
        answer_text += decoded_token
        next_token_embedding = get_embedding_for_token(model, tokenizer, decoded_token)
        next_token_embedding = next_token_embedding.to(embeddings.device)
        embeddings = torch.cat([embeddings, next_token_embedding], dim=1)
    
    # Extract and check answer
    predicted_answer = extract_numeric_answer(answer_text)
    
    try:
        ground_truth_numeric = float(ground_truth_answer)
    except:
        ground_truth_numeric = None
    
    is_correct = (predicted_answer is not None and 
                 ground_truth_numeric is not None and 
                 abs(predicted_answer - ground_truth_numeric) < 1e-6)
    
    return {
        'question': question,
        'ground_truth': ground_truth_answer,
        'predicted_answer': predicted_answer,
        'is_correct': is_correct,
        'reasoning': ''.join(reasoning_tokens),
        'answer_text': answer_text,
        'num_reasoning_steps': len(reasoning_tokens),
        'avg_entropy': float(sum(entropies) / len(entropies)) if entropies else 0.0
    }

def evaluate_on_gsm8k(num_examples=5, split="test", save_results=True, verbose=False, 
                      tokenizer=None, model=None, questions=None, answers=None, **kwargs):
    """Evaluate the model on a subset of GSM8K dataset - Sequential processing"""
    
    # Use provided questions/answers or load dataset
    if questions is None or answers is None:
        dataset = load_dataset("gsm8k", "main")
        examples = dataset[split][:num_examples]
        questions = examples["question"]
        answers = examples["answer"]
        answers = [answer.split("####")[1].strip() for answer in answers]
    
    results = []
    correct_count = 0
    total_time = 0
    
    print(f"Evaluating on {len(questions)} examples from {split} split...")
    
    # Process questions sequentially
    for i, (question, answer) in enumerate(tqdm(zip(questions, answers), 
                                               total=len(questions), 
                                               desc="Processing questions")):
        if verbose:
            print(f"\n--- Question {i+1} ---")
            print(f"Q: {question}")
            print(f"Expected: {answer}")
        
        start_time = time.time()
        
        try:
            result = process_single_question(
                question, answer, tokenizer, model, 
                verbose=verbose, **kwargs
            )
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
                
            if verbose:
                print(f"Predicted: {result['predicted_answer']}")
                print(f"Correct: {result['is_correct']}")
                
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            results.append({
                'question': question,
                'ground_truth': answer,
                'predicted_answer': None,
                'is_correct': False,
                'error': str(e)
            })
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if verbose:
            print(f"Time: {elapsed:.2f}s")
        
    # Calculate metrics
    accuracy = correct_count / len(results) if results else 0
    avg_time_per_question = total_time / len(results) if results else 0
    
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
    print(f"Average time per question: {avg_time_per_question:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    
    if save_results:
        with open(f"gsm8k_results_{split}_{len(questions)}.json", "w") as f:
            json.dump({
                'metadata': {
                    'num_examples': len(questions),
                    'split': split,
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_examples': len(results),
                    'avg_time_per_question': avg_time_per_question,
                    'total_time': total_time
                },
                'results': results
            }, f, indent=2)
        
        print(f"Results saved to gsm8k_results_{split}_{len(questions)}.json")
    
    # Free up any cached memory to reduce the risk of CUDA OOM errors between combinations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results, accuracy

def run_hyperparameter_grid_search(num_examples=10, split="test", verbose=False, 
                                  tokenizer=None, model=None, directory_name=None):
    """Run grid search over different hyperparameter combinations"""
    
    # Load the dataset once and select the questions to use consistently
    print(f"Loading {num_examples} questions from {split} split...")
    dataset = load_dataset("gsm8k", "main")
    examples = dataset[split][:num_examples]
    questions = examples["question"]
    answers = examples["answer"]
    answers = [answer.split("####")[1].strip() for answer in answers]
    
    print(f"Selected {len(questions)} questions for evaluation across all hyperparameter combinations")
    
    # Define parameter grids
    temperatures = [0.6]
    num_paths_list = [1, 2, 4, 8, 16]
    # Reduced the default search space to avoid excessive GPU memory usage
    num_samples_list = [32000]
    tau_thresholds = [1e-4]
    step_limits = [31000]
    
    all_results = []
    total_combinations = len(temperatures) * len(num_paths_list) * len(num_samples_list) * len(tau_thresholds) * len(step_limits)
    
    print(f"Running grid search with {total_combinations} combinations...")
    print(f"Parameters:")
    print(f"  Temperatures: {temperatures}")
    print(f"  Num paths: {num_paths_list}")
    print(f"  Num samples: {num_samples_list}")
    print(f"  Tau thresholds: {tau_thresholds}")
    print(f"  Step limits: {step_limits}")
    print(f"  Examples per combination: {num_examples}")
    
    combination_idx = 0
    
    for temperature in temperatures:
        for num_paths in num_paths_list:
            for num_samples in num_samples_list:
                for tau_threshold in tau_thresholds:
                    for step_limit in step_limits:
                        combination_idx += 1
                        
                        print(f"\n=== Combination {combination_idx}/{total_combinations} ===")
                        print(f"Temperature: {temperature}, Num paths: {num_paths}, "
                              f"Num samples: {num_samples}, Tau: {tau_threshold}, Step limit: {step_limit}")
                        
                        start_time = time.time()
                        
                        # Run evaluation with these parameters using the same questions
                        results, accuracy = evaluate_on_gsm8k(
                            num_examples=num_examples,
                            split=split,
                            save_results=False,
                            verbose=verbose,
                            tokenizer=tokenizer,
                            model=model,
                            questions=questions,  # Pass the same questions
                            answers=answers,      # Pass the same answers
                            num_paths=num_paths,
                            num_samples=num_samples,
                            tau_threshold=tau_threshold,
                            temperature=temperature,
                            step_limit=step_limit
                        )
                        
                        elapsed_time = time.time() - start_time
                        
                        # Free up any cached memory to reduce the risk of CUDA OOM errors between combinations
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        combination_result = {
                            'hyperparameters': {
                                'temperature': temperature,
                                'num_paths': num_paths,
                                'num_samples': num_samples,
                                'tau_threshold': tau_threshold,
                                'step_limit': step_limit
                            },
                            'metrics': {
                                'accuracy': accuracy,
                                'correct_count': sum(1 for r in results if r['is_correct']),
                                'total_examples': len(results),
                                'avg_time_per_question': elapsed_time / len(results) if results else 0,
                                'total_time': elapsed_time
                            },
                            'detailed_results': results
                        }
                        
                        all_results.append(combination_result)
                        
                        # Save individual combination results
                        filename = f"{directory_name}/gsm8k_results_t{temperature}_p{num_paths}_s{num_samples}_tau{tau_threshold}_sl{step_limit}_{split}_{num_examples}.json"
                        with open(filename, "w") as f:
                            json.dump(combination_result, f, indent=2)
                        print(f"Saved to {filename}")
    
    # Save comprehensive results
    summary_results = {
        'metadata': {
            'total_combinations': total_combinations,
            'num_examples_per_combination': num_examples,
            'split': split,
            'questions_used': questions,  # Store the actual questions used
            'answers_used': answers,      # Store the actual answers used
            'parameter_grids': {
                'temperatures': temperatures,
                'num_paths_list': num_paths_list,
                'num_samples_list': num_samples_list,
                'tau_thresholds': tau_thresholds,
                'step_limits': step_limits
            }
        },
        'all_combinations': all_results
    }
    
    summary_filename = f"{directory_name}/gsm8k_grid_search_summary_{split}_{num_examples}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\n=== GRID SEARCH COMPLETE ===")
    print(f"Total combinations tested: {total_combinations}")
    print(f"Results saved to {summary_filename}")
    
    # Find best combination
    best_result = max(all_results, key=lambda x: x['metrics']['accuracy'])
    best_params = best_result['hyperparameters']
    best_accuracy = best_result['metrics']['accuracy']
    
    print(f"\nBest combination:")
    print(f"  Temperature: {best_params['temperature']}")
    print(f"  Num paths: {best_params['num_paths']}")
    print(f"  Num samples: {best_params['num_samples']}")
    print(f"  Tau threshold: {best_params['tau_threshold']}")
    print(f"  Step limit: {best_params['step_limit']}")
    print(f"  Accuracy: {best_accuracy:.2%}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")

    parser.add_argument("--num-examples", type=int, default=50, 
                       help="Number of examples to evaluate")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for vanilla evaluation")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")

    args = parser.parse_args()
    print("Loading tokenizer...")
    model_name = args.model_name
    directory_name = f"./logs/{model_name}/{datetime.now().strftime('%Y-%m-%d')}"
    ## make directory if it doesn't exist
    os.makedirs(directory_name, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    # Set pad token if not already set (needed for batched generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="./offload",
        trust_remote_code=True
    )
    
    print(f"Model's distribution over the GPUs: {model.hf_device_map}")
    
    # Set model to evaluation mode
    model.eval()

    

    print("\n=== FULL EXPERIMENT ===")
    run_hyperparameter_grid_search(
        num_examples=args.num_examples, 
        split="test", 
        verbose=args.verbose,
        tokenizer=tokenizer,
        model=model,
        directory_name=directory_name
    )