import pytest
from src.preprocessing import validate_text

def test_valid_text_is_trimmed():
    assert validate_text("  legal clause  ") == "legal clause"

def test_empty_text_rejected():
    with pytest.raises(ValueError):
        validate_text(" ")

def test_long_text_rejected():
    with pytest.raises(ValueError):
        validate_text("x" * 20001)
