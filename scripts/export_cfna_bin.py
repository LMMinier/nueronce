#!/usr/bin/env python3
"""Export a microtorch CFNA checkpoint to the flat binary format read by the
C++ engine (cpp/cfna_engine.cpp).

Format (little-endian), version 1 — documented in cpp/README.md:

    magic   u32   0x414E4643 ("CFNA")
    version u32   1
    config  15 x i32   byte_embed_dim, d_local, d_model, p_max,
                       physical_blocks, logical_depth, n_heads, unit_window,
                       decoder_window, decoder_layers, d_state, channel_dim,
                       ret_byte_dim, min_patch, max_patch
    tau     f64
    trainable_segmentation u8
    n_tensors u32
    per tensor: ndim u32, dims u32[ndim], data f64[prod(dims)] (row-major)

Tensors are written in ``MicroCFNAModel.parameters()`` order, which is the
model's attribute-construction order (documented tensor-by-tensor in
cpp/README.md). float64 on purpose: the microtorch oracle computes in float64,
so the C++ engine can be parity-tested against it at ~1e-9 rather than the
looser tolerance a float32 export would force. A float32/int8 variant is
future work (documented, not claimed).

Usage:
    python scripts/export_cfna_bin.py checkpoints/micro_cfna_sft_100k/best.pt \
        --out cfna_model.bin
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from cfna.microtorch.chat import load_checkpoint

MAGIC = 0x414E4643  # "CFNA" little-endian
VERSION = 1
CONFIG_FIELDS = [
    "byte_embed_dim", "d_local", "d_model", "p_max", "physical_blocks",
    "logical_depth", "n_heads", "unit_window", "decoder_window",
    "decoder_layers", "d_state", "channel_dim", "ret_byte_dim",
    "min_patch", "max_patch",
]


def export(model, out_path: str) -> dict:
    cfg = model.cfg
    params = [p.data.astype(np.float64) for p in model.parameters()]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        for name in CONFIG_FIELDS:
            f.write(struct.pack("<i", int(getattr(cfg, name))))
        f.write(struct.pack("<d", float(cfg.tau)))
        f.write(struct.pack("<B", 1 if cfg.trainable_segmentation else 0))
        f.write(struct.pack("<I", len(params)))
        for arr in params:
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", d))
            f.write(np.ascontiguousarray(arr).tobytes())
    return {"n_tensors": len(params),
            "n_params": int(sum(a.size for a in params)),
            "bytes": Path(out_path).stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="cfna_model.bin")
    args = ap.parse_args()
    model, payload = load_checkpoint(args.checkpoint)
    info = export(model, args.out)
    fmt = (payload.get("meta") or {}).get("prompt_format", "legacy")
    print(f"exported {info['n_tensors']} tensors / {info['n_params']:,} params "
          f"-> {args.out} ({info['bytes']/1e6:.1f} MB, f64) | prompt_format={fmt}")


if __name__ == "__main__":
    main()
