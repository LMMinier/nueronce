# Nueronce Naming Migration

The repository public identity is now **Nueronce**, derived from *neuron* and
*nuance*.

## Canonical names

- Python package: `nueronce`
- Training/autograd runtime: `Nueronce Engine`
- Engine package: `nueronce.engine`
- Main model class: `NueronceModel`
- Architecture files: `nueronce_model.py` and `nueronce_blocks.py`
- Model family: `Nueronce 11M`, `Nueronce 35M`, `Nueronce 90M`, `Nueronce 1.2B`
- Checkpoints: `nueronce_35m_stepXXXX.pkl`

Legacy CFNA and MicroTorch terminology is not part of the public API.

## Validation performed

- Repository-wide legacy-name scan: zero remaining matches.
- Python bytecode compilation: passed.
- Initial 13 test modules: passed.
- Critical engine/model/SFT test group: no failures observed before the execution window ended.

Run the full validation locally with:

```bash
python -m pytest -q
```
