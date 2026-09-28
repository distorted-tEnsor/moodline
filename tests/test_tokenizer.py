"""The tokenizer replaced a pickled Keras object, so it is tested hardest."""

from __future__ import annotations

import numpy as np
import pytest

from app.tokenizer import TextEncoder, normalise


@pytest.fixture
def encoder() -> TextEncoder:
    vocab = {"<unk>": 1, "i": 2, "feel": 3, "happy": 4, "sad": 5, "rare": 9999}
    return TextEncoder(vocab, num_words=1000, max_length=6)


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("I feel HAPPY", "i feel happy"),
            ("I can't believe it!", "i cant believe it"),
            ("I can’t believe it", "i cant believe it"),
            ("hello,,,   world", "hello world"),
            ("  padded   out  ", "padded out"),
            ("emoji 😄 survives as space", "emoji survives as space"),
            ("numbers 42 stay", "numbers 42 stay"),
            ("", ""),
            ("!!!???", ""),
        ],
    )
    def test_cases(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected


class TestEncoding:
    def test_known_words_map_to_indices(self, encoder: TextEncoder) -> None:
        assert encoder.encode("i feel happy") == [2, 3, 4]

    def test_unknown_words_become_oov(self, encoder: TextEncoder) -> None:
        assert encoder.encode("i feel wobbly") == [2, 3, 1]

    def test_words_above_num_words_are_dropped_from_vocab(self) -> None:
        # Keras maps an index >= num_words back to the OOV token.
        encoder = TextEncoder({"<unk>": 1, "kept": 2, "dropped": 50}, num_words=10, max_length=4)
        assert "dropped" not in encoder.vocab
        assert encoder.encode("kept dropped") == [2, 1]

    def test_empty_text_encodes_to_nothing(self, encoder: TextEncoder) -> None:
        assert encoder.encode("   ") == []
        assert encoder.encode("!!!") == []


class TestPadding:
    def test_shape_and_post_padding(self, encoder: TextEncoder) -> None:
        tokens, lengths = encoder.encode_batch(["i feel happy", "i"])
        assert tokens.shape == (2, 6)
        assert tokens.dtype == np.int64
        np.testing.assert_array_equal(tokens[0], [2, 3, 4, 0, 0, 0])
        np.testing.assert_array_equal(tokens[1], [2, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(lengths, [3, 1])

    def test_truncates_from_the_end(self, encoder: TextEncoder) -> None:
        tokens, lengths = encoder.encode_batch(["i feel happy i feel happy i feel"])
        np.testing.assert_array_equal(tokens[0], [2, 3, 4, 2, 3, 4])
        assert lengths[0] == 6

    def test_pre_padding_option(self) -> None:
        encoder = TextEncoder({"<unk>": 1, "a": 2}, num_words=10, max_length=3, padding="pre")
        tokens, _ = encoder.encode_batch(["a"])
        np.testing.assert_array_equal(tokens[0], [0, 0, 2])

    def test_empty_text_is_all_padding_but_length_one(self, encoder: TextEncoder) -> None:
        tokens, lengths = encoder.encode_batch([""])
        assert tokens.sum() == 0
        assert lengths[0] == 1  # packing a zero-length sequence is undefined


class TestCoverage:
    def test_all_known(self, encoder: TextEncoder) -> None:
        assert encoder.coverage("i feel happy") == 1.0

    def test_half_known(self, encoder: TextEncoder) -> None:
        assert encoder.coverage("i wobbly") == 0.5

    def test_no_words(self, encoder: TextEncoder) -> None:
        assert encoder.coverage("!!!") == 0.0
