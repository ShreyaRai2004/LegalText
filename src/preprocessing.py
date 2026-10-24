"""Small text validation helpers. Preserve legal wording and punctuation."""
def validate_text(text: str, min_chars: int = 5, max_chars: int = 20000) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be text.")
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        raise ValueError(f"Please enter at least {min_chars} characters.")
    if len(cleaned) > max_chars:
        raise ValueError(f"Text is too long. Maximum length is {max_chars} characters.")
    return cleaned
