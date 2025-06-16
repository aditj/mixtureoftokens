import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
import json
import os
import sys

# Add parent directory to path to import from simple.py
sys.path.append('..')

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_name = "Qwen/Qwen3-4B"

def _sample_token_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> int:
    """Sample a single token ID from the given logits using temperature and top-p (nucleus) sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        probs_to_keep = sorted_probs.clone()
        probs_to_keep[sorted_indices_to_remove] = 0.0

        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(dim=-1, index=sorted_indices, src=probs_to_keep)

        probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

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
    """Generate text using a two-phase approach with embedding mixture."""
    device = model.device
    
    # Get embedding matrix for efficient lookup
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight
    
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
        last_logits = outputs.logits[:, -1, :].squeeze(0)
        scaled_logits = last_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Get top-k tokens and their probabilities
        if "nucleus" in experiment_name:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            top_k_probs, top_k_indices = sorted_probs[:torch.searchsorted(cum_probs, top_p)+1], sorted_indices[:torch.searchsorted(cum_probs, top_p)+1]
        else:
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            
        # Check stopping condition
        if end_token_id is not None and end_token_id in top_k_indices:
            end_token_pos = (top_k_indices == end_token_id).nonzero(as_tuple=True)[0]
            if len(end_token_pos) > 0:
                end_prob = top_k_probs[end_token_pos[0]]
                if end_prob >= min_end_prob:
                    print(f"Stopping Phase 1 at round {round_idx} due to end token probability: {end_prob:.3f}")
                    break
        
        # Get embeddings for top-k tokens
        top_k_embeddings = embedding_matrix[top_k_indices]
        
        # Create weighted mixture of embeddings
        if "element_wise_max" in experiment_name:
            mixed_embedding = torch.max(top_k_embeddings, dim=0).values.unsqueeze(0).unsqueeze(0)
        else:
            if "inverse_p" in experiment_name:
                normalized_probs = 1 / top_k_probs
            else:
                normalized_probs = top_k_probs / top_k_probs.sum()
                
            mixed_embedding = torch.sum(
                top_k_embeddings * normalized_probs.unsqueeze(-1), 
                dim=0
            ).unsqueeze(0).unsqueeze(0)
            
        if "dirichlet" in experiment_name:
            d = torch.distributions.dirichlet.Dirichlet(normalized_probs)
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
        
        # For tracking purposes, sample one token from the mixture
        if "element_wise_max" not in experiment_name:
            sampled_token_id = torch.multinomial(normalized_probs, num_samples=1).item()
            actual_token_id = top_k_indices[sampled_token_id].item()
        else:
            # For element-wise max, just take the highest probability token
            actual_token_id = top_k_indices[0].item()
            normalized_probs = top_k_probs / top_k_probs.sum()
            
        generated_token_ids.append(actual_token_id)
        phase1_tokens.append((top_k_indices.tolist(), normalized_probs.tolist()))
        phase1_rounds_completed += 1
    
    # Add </think> token
    think_end_token = "</think>"
    try:
        think_end_ids = tokenizer.encode(think_end_token, add_special_tokens=False)
        for token_id in think_end_ids:
            generated_token_ids.append(token_id)
            transition_tokens.append(token_id)
            
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
    
    # Set temperature for phase 2
    phase2_temperature = 0.6
    
    # Add <think> or <answer> token to start phase 2
    if PHASE_2_STRATEGY == "think_first":
        think_start_token = "<think>"
    else:
        think_start_token = "<answer>"
        
    try:
        think_start_ids = tokenizer.encode(think_start_token, add_special_tokens=False)
        for token_id in think_start_ids:
            generated_token_ids.append(token_id)
            transition_tokens.append(token_id)
            
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
        print(f"Warning: Could not add {think_start_token} token")
    
    # -------------------------------------------------------------------------
    # Phase 2: Standard token-by-token generation for T_exp rounds
    # -------------------------------------------------------------------------
    print(f"Phase 2: Standard generation for {T_exp} rounds...")
    
    for _ in range(T_exp):
        last_logits = outputs.logits[:, -1, :].squeeze(0)
        next_token_id = _sample_token_from_logits(last_logits, temperature=phase2_temperature)
        
        generated_token_ids.append(next_token_id)
        phase2_tokens.append(next_token_id)
        
        if next_token_id == tokenizer.eos_token_id:
            break
        
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

def load_model():
    """Load the model and tokenizer globally."""
    global model, tokenizer
    if model is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded successfully!")

@app.route('/')
def index():
    """Serve the main interface."""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """Handle generation requests."""
    try:
        data = request.json
        
        # Extract parameters
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        user_prompt = data.get('user_prompt', 'Hello!')
        k = int(data.get('k', 5))
        T_e = int(data.get('T_e', 50))
        T_total = int(data.get('T_total', 250))
        temperature = float(data.get('temperature', 0.8))
        experiment_name = data.get('experiment_name', 'non_uniform')
        
        T_exp = T_total - T_e
        
        # Build the chat template prompt
        prompt_text = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Determine phase 2 strategy from experiment name
        if "answer_directly" in experiment_name:
            PHASE_2_STRATEGY = "answer_first"
        else:
            PHASE_2_STRATEGY = "think_first"

        # Generate with embedding mixture
        generated_text, phase_info = generate_with_embedding_mixture(
            model,
            tokenizer,
            prompt_text,
            T_e=T_e,
            T_exp=T_exp,
            k=k,
            temperature=temperature,
            min_end_prob=0.05,
            return_phase_info=True,
            PHASE_2_STRATEGY=PHASE_2_STRATEGY,
            experiment_name=experiment_name,
        )
        
        # Pre-fetch all token texts to avoid multiple API calls
        token_texts = {}
        
        # Collect all unique token IDs from phase 1
        for round_tokens, _ in phase_info['phase1_token_ids']:
            for token_id in round_tokens:
                if token_id not in token_texts:
                    token_texts[token_id] = tokenizer.decode(token_id)
        
        # Collect all token IDs from phase 2  
        for token_id in phase_info['phase2_token_ids']:
            if token_id not in token_texts:
                token_texts[token_id] = tokenizer.decode(token_id)
                
        # Collect all transition token IDs
        for token_id in phase_info['transition_token_ids']:
            if token_id not in token_texts:
                token_texts[token_id] = tokenizer.decode(token_id)
        
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'phase_info': phase_info,
            'token_texts': token_texts,  # Include all token texts
            'prompt_used': prompt_text,
            'parameters': {
                'k': k,
                'T_e': T_e,
                'T_total': T_total,
                'temperature': temperature,
                'experiment_name': experiment_name
            }
        })
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    


@app.route('/api/token-text/<int:token_id>')
def get_token_text(token_id):
    """Get the text representation of a token ID."""
    try:
        token_text = tokenizer.decode(token_id)
        return jsonify({
            'token_id': token_id,
            'token_text': token_text
        })
    except Exception as e:
        return jsonify({
            'token_id': token_id,
            'token_text': f'Token_{token_id}',
            'error': str(e)
        })

@app.route('/api/model-status')
def model_status():
    """Check if model is loaded."""
    return jsonify({
        'loaded': model is not None,
        'model_name': model_name if model is not None else None
    })

if __name__ == '__main__':
    load_model()
    
    app.run(debug=True, host='0.0.0.0', port=8000) 