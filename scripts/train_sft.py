#!/usr/bin/env python3
"""Supervised instruction tuning (VGRFT stage 1) on top of a byte-LM checkpoint.

Every checkpoint produced by ``train_checkpoint.py`` is trained with pure
next-byte cross-entropy on monologic prose — it learns what English looks like,
not what to say back when spoken to (see ``cfna/chat.py``'s honest framing).
This script closes that gap for real, with three backends:

- ``--backend torch --model small`` (default, backward-compatible): fine-tunes
  the real PyTorch ``CFNAModel`` on the small 61-turn hand-written dialogue set.
- ``--backend microtorch --model small``: the same small dialogue set, but on
  the from-scratch microtorch engine (``MicroByteLM``), no PyTorch needed.
- ``--backend microtorch --model full-cfna``: the real, large-scale run —
  continuous sharded SFT of the full ``MicroCFNAModel`` over a JSONL dialogue
  corpus (e.g. 10 shards x 10,000 conversations), with resumable
  shard/step/optimizer state, checkpointing, LR decay, and validation/test
  evaluation. See ``cfna.training.sharded_sft``.

Usage (small, backward-compatible):
    python scripts/train_sft.py --ckpt checkpoints/cfna_chat.pt \\
        --out checkpoints/cfna_chat_sft.pt --steps 400

Usage (large-scale, microtorch, full architecture):
    python scripts/train_sft.py \\
        --backend microtorch --model full-cfna \\
        --train-dir data/sft_100k/train_shards \\
        --validation data/sft_100k/validation.jsonl \\
        --test data/sft_100k/test.jsonl \\
        --num-shards 10 --examples-per-shard 10000 \\
        --save-dir checkpoints/micro_cfna_sft_100k \\
        --resume --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEMO_TURNS = ["Hello, who are you?", "What is two plus two?", "Thank you, goodbye."]


def run_torch_small(args):
    import torch

    from cfna.chat import Conversation, load_checkpoint
    from cfna.model import CFNAModel, ModelConfig
    from cfna.training.sft import SFT_DATASET, held_out_split, sft_eval, make_sft_batch, TorchSFTBackend
    from cfna.training.vgrft import VGRFTTrainer

    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        model, ckpt = load_checkpoint(str(ckpt_path))
        print(f"loaded pretrained checkpoint {ckpt_path} ({model.num_params():,} params)")
    else:
        model = CFNAModel(ModelConfig())
        ckpt = {"config": vars(model.cfg)}
        print(f"no checkpoint at {ckpt_path}; starting SFT from a freshly-initialized "
              f"({model.num_params():,}-param) model — pretrain first for a real result")

    train_ex, val_ex = held_out_split(SFT_DATASET, val_frac=args.val_frac, seed=args.seed)
    print(f"SFT set: {len(train_ex)} train turns / {len(val_ex)} held-out turns\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    trainer = VGRFTTrainer(TorchSFTBackend(model, opt))
    history = trainer.supervised_instruction_tune(
        train_ex, steps=args.steps, batch_size=args.batch, val_examples=val_ex,
        seed=args.seed, log_every=max(1, args.steps // 10),
    )

    for rec in history:
        line = f"step {rec['step']:5d} | train loss {rec['train_loss']:.3f}"
        if "val_loss" in rec:
            line += f" | held-out loss {rec['val_loss']:.3f}"
        print(line)

    val_batch = make_sft_batch(val_ex)
    final_val = sft_eval(model, val_batch)
    print(f"\nfinal held-out SFT loss (response bytes only): {final_val:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(model.cfg),
                "step": ckpt.get("step", 0), "history": ckpt.get("history", []),
                "sft_history": history}, out)
    print(f"saved SFT-tuned checkpoint -> {out}")

    print("\n=== sample turns after SFT ===")
    convo = Conversation(model, temperature=0.0, max_new=60)
    for turn in DEMO_TURNS:
        reply = convo.say(turn)
        print(f"User:      {turn}")
        print(f"Assistant: {reply}\n")


def run_microtorch_small(args):
    import numpy as np

    from cfna.microtorch.models import MicroByteLM, MicroSFTBackend
    from cfna.training.dialogue_data import SFT_DATASET, held_out_split
    from cfna.training.vgrft import VGRFTTrainer

    np.random.seed(args.seed)
    model = MicroByteLM(d_model=32, n_heads=4, window=16, d_state=8)
    print(f"microtorch MicroByteLM: {sum(p.data.size for p in model.parameters()):,} params")

    train_ex, val_ex = held_out_split(SFT_DATASET, val_frac=args.val_frac, seed=args.seed)
    print(f"SFT set: {len(train_ex)} train turns / {len(val_ex)} held-out turns\n")

    trainer = VGRFTTrainer(MicroSFTBackend(model, lr=args.lr))
    history = trainer.supervised_instruction_tune(
        train_ex, steps=args.steps, batch_size=args.batch, val_examples=val_ex,
        seed=args.seed, log_every=max(1, args.steps // 10),
    )
    for rec in history:
        line = f"step {rec['step']:5d} | train loss {rec['train_loss']:.3f}"
        if "val_loss" in rec:
            line += f" | held-out loss {rec['val_loss']:.3f}"
        print(line)


def run_microtorch_full_cfna(args):
    from cfna.microtorch.cfna_model import MicroModelConfig
    from cfna.training.sharded_sft import ShardedSFTConfig, run_sharded_sft

    if not args.train_dir or not args.validation or not args.test:
        raise SystemExit("--backend microtorch --model full-cfna requires --train-dir, "
                          "--validation, and --test")

    # Deliberately unchanged from the small demo config (~112K params) — this
    # run tests whether more data helps *this* architecture, not a bigger one.
    model_cfg = MicroModelConfig(
        byte_embed_dim=16, d_local=24, d_model=32, p_max=16, physical_blocks=1,
        logical_depth=2, n_heads=4, unit_window=12, decoder_window=16,
        decoder_layers=1, d_state=8, channel_dim=8, ret_byte_dim=8,
        min_patch=2, max_patch=14,
    )
    cfg = ShardedSFTConfig(
        train_dir=args.train_dir, val_path=args.validation, test_path=args.test,
        save_dir=args.save_dir, metrics_dir=args.metrics_dir,
        num_shards=args.num_shards, examples_per_shard=args.examples_per_shard,
        batch_size=args.batch, max_len=args.max_len, lr=args.lr,
        lr_decay_factor=args.lr_decay_factor, min_lr=args.min_lr,
        grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps,
        periodic_val_every=args.periodic_val_every, periodic_val_examples=args.periodic_val_examples,
        full_val_examples=args.full_val_examples, checkpoint_every_steps=args.checkpoint_every_steps,
        log_every=args.log_every, seed=args.seed, resume=args.resume,
    )
    summary = run_sharded_sft(model_cfg, cfg)
    print(f"\ndone: best shard {summary['best_shard']} (val loss {summary['best_val_loss']:.4f}) | "
          f"test loss {summary['test_metrics']['loss']:.4f} "
          f"(bits/byte {summary['test_metrics']['bits_per_byte']:.4f}, "
          f"byte acc {summary['test_metrics']['byte_accuracy']:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "microtorch"], default="torch")
    ap.add_argument("--model", choices=["small", "full-cfna"], default="small")

    # small-scale (torch or microtorch) args
    ap.add_argument("--ckpt", type=str, default="checkpoints/cfna_chat.pt")
    ap.add_argument("--out", type=str, default="checkpoints/cfna_chat_sft.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--val-frac", type=float, default=0.2)

    # large-scale (microtorch full-cfna) args
    ap.add_argument("--train-dir", type=str, default=None, help="directory of shard_NN.jsonl files")
    ap.add_argument("--validation", type=str, default=None)
    ap.add_argument("--test", type=str, default=None)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--examples-per-shard", type=int, default=10_000)
    ap.add_argument("--save-dir", type=str, default="checkpoints/micro_cfna_sft_100k")
    ap.add_argument("--metrics-dir", type=str, default="metrics")
    ap.add_argument("--max-len", type=int, default=288)
    ap.add_argument("--lr-decay-factor", type=float, default=0.7)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--periodic-val-every", type=int, default=200)
    ap.add_argument("--periodic-val-examples", type=int, default=256)
    ap.add_argument("--full-val-examples", type=int, default=None)
    ap.add_argument("--checkpoint-every-steps", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")

    # shared
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.backend == "torch" and args.model == "small":
        run_torch_small(args)
    elif args.backend == "microtorch" and args.model == "small":
        run_microtorch_small(args)
    elif args.backend == "microtorch" and args.model == "full-cfna":
        run_microtorch_full_cfna(args)
    else:
        raise SystemExit(f"unsupported combination: --backend {args.backend} --model {args.model} "
                          f"(torch + full-cfna is not implemented; use microtorch for full-cfna)")


if __name__ == "__main__":
    main()
