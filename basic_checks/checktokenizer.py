# check how the qwen2.5 tokenizer tokenizes </think> token

from transformers import AutoTokenizer

def check_tokenization():
    # Load Qwen2.5 tokenizer
    print("Loading Qwen2.5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    # Text to tokenize
    text = "</think>"
    
    print(f"\nTokenizing: '{text}'")
    print("-" * 50)
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Show results
    print(f"Original text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Decode back to verify
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded back: '{decoded}'")
    
    # Show individual token details
    print("\nToken details:")
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        print(f"  {i}: '{token}' -> ID: {token_id}")
    
    # Also check with special tokens (just in case)
    print("\n" + "="*50)
    print("With special tokens:")
    token_ids_with_special = tokenizer.encode(text, add_special_tokens=True)
    print(f"Token IDs (with special tokens): {token_ids_with_special}")
    decoded_with_special = tokenizer.decode(token_ids_with_special)
    print(f"Decoded (with special tokens): '{decoded_with_special}'")

if __name__ == "__main__":
    check_tokenization()
