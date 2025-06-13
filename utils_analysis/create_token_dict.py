#!/usr/bin/env python3
"""
Create a token dictionary for the dashboard visualization.
This script loads the tokenizer and creates a mapping from token IDs to their text representations.
"""

from transformers import AutoTokenizer
import json
import argparse

def create_token_dict(model_name="Qwen/Qwen2.5-3B-Instruct", output_format="json"):
    """
    Create a dictionary mapping token IDs to their text representations.
    
    Args:
        model_name: Name of the model/tokenizer to use
        output_format: Format to save the dictionary ('json' or 'js')
    
    Returns:
        dict: Dictionary mapping token IDs to text
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Creating token dictionary...")
    token_dict = {}
    
    # Get the vocabulary size
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create mapping for all tokens in vocabulary
    for token_id in range(vocab_size):
        try:
            # Convert token ID to text
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            
            # Clean up the token text for better display
            # Remove leading/trailing whitespace but preserve internal structure
            if token_text.strip():
                token_dict[token_id] = token_text
            else:
                # For whitespace-only tokens, show them explicitly
                if token_text == " ":
                    token_dict[token_id] = "⎵"  # Visible space character
                elif token_text == "\n":
                    token_dict[token_id] = "↵"  # Visible newline character
                elif token_text == "\t":
                    token_dict[token_id] = "→"  # Visible tab character
                else:
                    token_dict[token_id] = f"[{repr(token_text)}]"
                    
        except Exception as e:
            # Some token IDs might not be valid
            token_dict[token_id] = f"[Invalid Token {token_id}]"
    
    print(f"Created dictionary with {len(token_dict)} tokens")
    
    return token_dict

def save_token_dict(token_dict, filename, format_type="json"):
    """
    Save the token dictionary in the specified format.
    
    Args:
        token_dict: Dictionary mapping token IDs to text
        filename: Output filename
        format_type: 'json' or 'js'
    """
    if format_type == "json":
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(token_dict, f, indent=2, ensure_ascii=False)
        print(f"Token dictionary saved as JSON: {filename}")
    
    elif format_type == "js":
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("// Token dictionary for dashboard\n")
            f.write("// Replace the tokenDict assignment in your dashboard with this:\n\n")
            f.write("const tokenDict = ")
            json.dump(token_dict, f, indent=2, ensure_ascii=False)
            f.write(";\n\n")
            f.write("// Usage: Copy the tokenDict object above and paste it into your dashboard HTML file\n")
        print(f"Token dictionary saved as JavaScript: {filename}")

def print_sample_tokens(token_dict, num_samples=20):
    """Print a sample of tokens for verification."""
    print(f"\nSample tokens (first {num_samples}):")
    print("-" * 50)
    
    for i, (token_id, token_text) in enumerate(list(token_dict.items())[:num_samples]):
        print(f"Token {token_id:5d}: '{token_text}'")
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Create token dictionary for dashboard visualization")
    parser.add_argument('--model', default="Qwen/Qwen2.5-3B-Instruct", 
                       help='Model name for tokenizer (default: Qwen/Qwen2.5-3B-Instruct)')
    parser.add_argument('--output', default="token_dict.json",
                       help='Output filename (default: token_dict.json)')
    parser.add_argument('--format', choices=['json', 'js'], default='json',
                       help='Output format: json or js (default: json)')
    parser.add_argument('--sample', action='store_true',
                       help='Show sample tokens after creation')
    
    args = parser.parse_args()
    
    # Create the token dictionary
    token_dict = create_token_dict(args.model, args.format)
    
    # Save the dictionary
    save_token_dict(token_dict, args.output, args.format)
    
    # Show samples if requested
    if args.sample:
        print_sample_tokens(token_dict)
    
    print(f"\nDictionary creation complete!")
    
    if args.format == "js":
        print("\nTo use in your dashboard:")
        print("1. Open the generated .js file")
        print("2. Copy the tokenDict object")
        print("3. Replace the tokenDict in your dashboard HTML file")
    else:
        print("\nTo use in your dashboard:")
        print("1. Load this JSON file in your dashboard")
        print("2. Or convert to JavaScript format using --format js")

if __name__ == "__main__":
    main() 