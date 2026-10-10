import pytest

from kv_cache_orchestrator.controller import parse_byte_size


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1.9, 1),
        ("0B", 0),
        (" 1kb ", 1024),
        ("1.5M", 1572864),
        ("1T", 1024**4),
        ("1\u2003KB", 1024),
    ],
)
def test_parse_byte_size_accepts_numeric_values_and_unit_boundaries(value, expected):
    assert parse_byte_size(value) == expected


@pytest.mark.parametrize("value", ["", " ", "1P", "-1B", "１KB"])
def test_parse_byte_size_rejects_empty_unsupported_and_unicode_digit_values(value):
    with pytest.raises(ValueError, match="invalid byte size"):
        parse_byte_size(value)
