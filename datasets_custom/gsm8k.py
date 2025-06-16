import re
import json

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