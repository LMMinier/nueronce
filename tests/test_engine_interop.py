import numpy as np
import torch

from nueronce.engine.interop import load_torch_state_dict
from nueronce.engine.nueronce_model import NueronceConfig, NueronceModel
from nueronce.model import ModelConfig, NUERONCEModel


def test_torch_checkpoint_maps_completely_and_preserves_predictions():
    cfg = dict(byte_embed_dim=8, d_local=12, d_model=16, p_max=8,
               physical_blocks=1, logical_depth=1, n_heads=2, unit_window=8,
               decoder_window=8, decoder_layers=1, d_state=4, channel_dim=4,
               ret_byte_dim=4, min_patch=2, max_patch=8)
    torch.manual_seed(3)
    torch_model = NUERONCEModel(ModelConfig(**cfg)).eval()
    engine_model = NueronceModel(NueronceConfig(**cfg))
    report = load_torch_state_dict(engine_model, torch_model.state_dict())
    assert report["loaded"] == len(torch_model.state_dict())

    ids = np.random.default_rng(4).integers(0, 256, size=(1, 32), dtype=np.int64)
    with torch.no_grad():
        torch_logits, _ = torch_model(torch.from_numpy(ids))
    engine_logits, _ = engine_model(ids)
    assert np.array_equal(torch_logits.numpy().argmax(-1), engine_logits.data.argmax(-1))
