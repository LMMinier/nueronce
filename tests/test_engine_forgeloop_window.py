import numpy as np

from scripts.train_forgeloop_engine_sft import target_window


def test_target_window_preserves_response_targets():
    row = {"prompt": "p" * 500, "response": "answer"}
    ids, mask = target_window(row, "system", 64)
    assert ids.shape == mask.shape == (1, 64)
    assert np.any(mask[:, 1:])
    assert not np.any(mask[:, :32])
    assert ids[mask].astype(np.uint8).tobytes()
