from scripts.probe_base_continuation_engine import repetition_stats


def test_repetition_stats_flags_short_cycle():
    looped = repetition_stats(b" the the the the")
    prose = repetition_stats(b"A quick fox jumps.")
    assert looped["repeated_ngram_fraction"] > 0.5
    assert prose["repeated_ngram_fraction"] == 0.0
