"""Config presets: parameter counts verified by construction (engine), and
field parity between the torch and engine preset definitions."""

import numpy as np
import pytest

from nueronce.engine.nueronce_model import NueronceModel, preset_configs

EXPECTED = {
    "chat_11m": (10_500_000, 11_800_000),
    "base_35m": (33_000_000, 36_000_000),
    "base_90m": (89_000_000, 95_000_000),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_preset_param_counts_in_documented_range(name):
    lo, hi = EXPECTED[name]
    n = NueronceModel(preset_configs()[name]).num_params()
    assert lo < n < hi, f"{name}: {n:,} params outside documented range"


def test_torch_and_engine_presets_agree_field_for_field():
    torch_side = pytest.importorskip("nueronce.model", reason="needs torch")
    micro = preset_configs()
    for name, factory in torch_side.CONFIG_PRESETS.items():
        if name == "large_337m":
            continue
        tcfg = vars(factory())
        mcfg = vars(micro[name])
        assert tcfg == mcfg, f"{name}: preset drift between backends"
