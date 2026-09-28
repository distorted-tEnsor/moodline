"""Engine behaviour: correctness of the converted weights, caching, batching."""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_artifact

pytestmark = requires_artifact

REFERENCE_CASES = [
    ("i feel so alone and hopeless today", "sadness"),
    ("i am furious that they cancelled the trip", "anger"),
    ("i feel terrified walking home in the dark", "fear"),
    ("i was shocked and completely surprised by the unexpected gift", "surprise"),
    ("i am feeling romantic and tender tonight", "love"),
]


class TestPredictions:
    @pytest.mark.parametrize(("text", "expected"), REFERENCE_CASES)
    def test_known_sentences(self, engine, text, expected) -> None:
        prediction = engine.predict(text)
        assert prediction.label == expected
        assert prediction.probabilities[expected] > 0.4

    def test_probabilities_form_a_distribution(self, engine) -> None:
        prediction = engine.predict("i feel wonderful about everything")
        assert set(prediction.probabilities) == set(engine.labels)
        assert math.isclose(sum(prediction.probabilities.values()), 1.0, rel_tol=1e-5)
        assert all(0.0 <= value <= 1.0 for value in prediction.probabilities.values())

    def test_label_is_the_argmax(self, engine) -> None:
        prediction = engine.predict("everything is falling apart again")
        assert prediction.label == max(prediction.probabilities, key=prediction.probabilities.get)


class TestBatchInvariance:
    """A sentence must score the same alone as in company; padding bugs show here."""

    def test_same_scores_alone_and_in_a_batch(self, engine) -> None:
        texts = [text for text, _ in REFERENCE_CASES]
        alone = [engine.predict_many([text], use_cache=False)[0] for text in texts]
        together = engine.predict_many(texts, use_cache=False)

        for one, many in zip(alone, together, strict=True):
            assert one.label == many.label
            for label in engine.labels:
                assert one.probabilities[label] == pytest.approx(
                    many.probabilities[label], abs=1e-6
                )

    def test_order_is_preserved(self, engine) -> None:
        texts = ["i am furious", "i feel loved", "i am scared"]
        results = engine.predict_many(texts, use_cache=False)
        assert [result.text for result in results] == texts


class TestCache:
    def test_repeat_is_served_from_cache(self, engine) -> None:
        engine.clear_cache()
        first = engine.predict("a sentence that is only used in this test")
        second = engine.predict("a sentence that is only used in this test")
        assert first.cached is False
        assert second.cached is True
        assert second.probabilities == first.probabilities

    def test_cache_key_is_normalised(self, engine) -> None:
        engine.clear_cache()
        engine.predict("I can't stand it")
        assert engine.predict("i cant STAND it!!").cached is True

    def test_cache_can_be_disabled(self, settings) -> None:
        from app.engine import EmotionEngine

        engine = EmotionEngine.load(settings.model_copy(update={"cache_size": 0}))
        engine.predict("no cache here")
        assert engine.predict("no cache here").cached is False
        assert engine.cache_size == 0


class TestEdgeCases:
    def test_empty_and_punctuation_only_text(self, engine) -> None:
        for text in ("", "   ", "!!!???"):
            prediction = engine.predict(text)
            assert prediction.label in engine.labels
            assert prediction.tokens == 1  # all padding

    def test_pure_out_of_vocabulary_text(self, engine) -> None:
        prediction = engine.predict("xyzzy plugh frobnicate")
        assert prediction.coverage == 0.0
        assert prediction.label in engine.labels

    def test_very_long_text_is_truncated(self, engine) -> None:
        prediction = engine.predict("i feel happy " * 500)
        assert prediction.tokens == engine.encoder.max_length

    def test_unicode_does_not_explode(self, engine) -> None:
        assert engine.predict("je me sens 😄 très heureux — vraiment").label in engine.labels

    def test_empty_batch(self, engine) -> None:
        assert engine.predict_many([]) == []


class TestKnownWeakness:
    """`dair-ai/emotion` is heavily templated ("i feel X"), and the model
    inherited that. These assertions pin the *current* behaviour so a retrained
    model visibly improves on them rather than regressing unnoticed."""

    def test_off_template_phrasing_is_flagged_as_uncertain(self, engine) -> None:
        prediction = engine.predict("i can't believe how happy i am right now")
        # It gets this wrong today. What matters is that it does not claim confidence.
        assert prediction.probabilities[prediction.label] < 0.5

    def test_grateful_is_confidently_wrong(self, engine) -> None:
        """The worst case found so far, and the clearest argument for retraining.

        "grateful" is in vocabulary (index 735) and the model still calls it
        sadness at 0.99. This is inherited, not introduced: the original Keras
        graph returns the same probabilities to 1e-7. A retrained model should
        break this test.
        """
        prediction = engine.predict("i feel grateful")
        assert prediction.label == "sadness"
        assert prediction.probabilities["sadness"] > 0.9
        assert prediction.coverage == 1.0  # not an out-of-vocabulary artefact

    def test_love_collapses_into_joy(self, engine) -> None:
        # `love` is the smallest class and the model leans on a few cue words;
        # "affection" is not one of them.
        assert engine.predict("i feel so much affection for him").label == "joy"

    def test_on_template_phrasing_is_confident(self, engine) -> None:
        prediction = engine.predict("i feel happy and excited")
        assert prediction.label == "joy"
        assert prediction.probabilities["joy"] > 0.5


class TestIntrospection:
    def test_info_reports_the_essentials(self, engine) -> None:
        info = engine.info()
        assert info["parameters"] == 3_454_662
        assert info["vocab_size"] == 9_999
        assert info["max_sequence_length"] == 50
        assert engine.version.startswith("bigru-")
