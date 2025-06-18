

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
